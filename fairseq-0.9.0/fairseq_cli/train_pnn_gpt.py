#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random
import os
import numpy as np
import torch
import shutil
import time

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training
    """
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size == 1:
        return

    def _send_and_wait(r):
        if rank == r:
            tensor = torch.tensor(0, device="cuda")
        else:
            tensor = torch.tensor(1, device="cuda")
        torch.distributed.broadcast(tensor, r)
        while tensor.item() == 1:
            time.sleep(1)

    _send_and_wait(0)
    # now sync on the main process
    _send_and_wait(1)

def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)
    arch = args.arch.split('_')[1]

    data_dir = args.data.split(':')
    args.data_dir = data_dir

    memory_dir = args.memory_dir.split(':')
    args.memory_dir = memory_dir


    domains = args.domains.split(':')
    args.domains = domains
    args.domain_idx = {}
    for i,domain in enumerate(domains):
        args.domain_idx[domain] = i

    added_head_num = list(map(int, args.added_head_num.split(':')))
    added_layer_num = list(map(int, args.added_layer_num.split(':')))
    max_updates = list(map(int, args.max_update.split(':')))
    if args.max_train_times != None:
        args.max_train_times = list(map(int, args.max_train_times.split(':')))
    else:
        args.max_train_times = [1e9] * 10 



    models_distil_from_path = args.models_distil_from.split(':')
    models_distil_from = {}
    args.best_ppl = {}
    args.ppl_increased = []

    args.layer_candidates = None
    args.layer_idxs = None
    sm_arch = None

    path = os.path.join(args.save_dir,"checkpoint_0_WB.pt")
    trained_domains = []
    args.best_ppl = {}
    args.last_validation_time = 0
    args.valid_loss_list = {}
    args.trained_domains = []

    if os.path.exists(path):
        ckpt = torch.load(path)
        trained_domains = ckpt.get("trained_domains", [])
        args.trained_domains = trained_domains
        args.layer_candidates = ckpt['args'].layer_candidates
        args.layer_idxs = ckpt['args'].layer_idxs
        try:
            args.best_ppl = ckpt['args'].best_ppl
        except:
            pass
        try:
            args.last_validation_time = ckpt['args'].last_validation_time
        except:
            pass
        try:
            args.valid_loss_list = ckpt['args'].valid_loss_list
        except:
            pass

        print(args.layer_candidates)
        print(args.layer_idxs)
        print(args.best_ppl)
        del ckpt
    print("trained domains: " + str(trained_domains))

    for i in range(len(data_dir)):
        '''
        if i <= 0: args.reset_optimizer=False
        else: args.reset_optimizer=True
        '''
        synchronize()
        args.cur = i
        args.cur_domain = domains[i]
        args.cur_max_update = max_updates[i]
        args.max_update_distil = max_updates[i]
        print("train %s domain" % args.cur_domain)
        #args.memory_bank = args.save_dir + '/memory_bank_' + domains[i-1] +'.pt' if i > 0 else None

        args.restore_file = os.path.join(args.save_dir,"checkpoint_{}_{}.pt".format(i, domains[i]))

        if len(trained_domains) > 0 and (args.cur_domain in trained_domains):
            print("%s domain has been trained" % args.cur_domain)
            continue
        if i > 0:
            shutil.copyfile(os.path.join(args.save_dir,"checkpoint_{}_{}.pt".format(i-1, domains[i-1])), os.path.join(args.save_dir,"checkpoint_{}_{}.pt".format(i, domains[i])))
            shutil.copyfile(os.path.join(args.save_dir,"checkpoint_{}_{}.pt".format(i, domains[i])), os.path.join(args.save_dir,"checkpoint_last.pt"))

        synchronize()
        # Setup task, e.g., translation, language modeling, etc.
        task = tasks.setup_task(args)
        # Build model and criterion
        model = task.build_model(args)
        criterion = task.build_criterion(args)
        print(model)
        
        if arch == 'roberta':
            for k in range(len(args.domains)):
                if k != i:
                    for param in model.Roberta_dict[str(k)].parameters():
                        param.requires_grad = False
                    for j in range(k):
                        for l in range(args.encoder_layers + 1):
                            for param in model.Linear_dict[str(l) + '_' + str(j) + '_' + str(k)].parameters():
                                param.requires_grad = False
        if arch == 'gpt':
            for k in range(len(args.domains)):
                if k != i:
                    for param in model.GPT_dict[str(k)].parameters():
                        param.requires_grad = False
                    for j in range(k):
                        for l in range(args.decoder_layers + 1):
                            for param in model.Linear_dict[str(l) + '_' + str(j) + '_' + str(k)].parameters():
                                param.requires_grad = False

        print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
        num_para = sum(p.numel() for p in model.parameters())
        print('| num. model params: {} (num. trained: {})'.format(
            num_para,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        ))
            
        if args.replay:
            for domain in domains[:i]:
                task.load_memory(domain)

        # Build trainer
        trainer = Trainer(args, task, model, models_distil_from, criterion)
        print('| training on {} GPUs'.format(args.distributed_world_size))
        print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
            args.max_tokens,
            args.max_sentences,
        ))

        print("start training on %s domain" % domains[i])
        args.data = data_dir[i]

        try:
            extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, domain=domains[i])
        except:
            args.restore_file = os.path.join(args.save_dir,"checkpoint_last.pt")
            extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, domain=domains[i])
            args.restore_file = os.path.join(args.save_dir,"checkpoint_{}_{}.pt".format(i, domains[i]))

        # Train until the learning rate gets too small
        max_epoch = args.max_epoch or math.inf
        max_update = max_updates[i] or math.inf
        lr = trainer.get_lr()
        train_meter = StopwatchMeter()
        train_meter.start()
        valid_subsets = args.valid_subset.split(',')
        while (
            lr > args.min_lr
            and (epoch_itr.epoch < max_epoch or (epoch_itr.epoch == max_epoch
                and epoch_itr._next_epoch_itr is not None))
            and trainer.get_num_updates() < max_update
        ):
            # train for one epoch
            training_time = train(args, trainer, task, epoch_itr, cur_domain=domains[i])

            if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
                name = domains[i]
                if name not in args.valid_loss_list.keys():
                    args.valid_loss_list[name] = []
                args.ppl_increased = []
                valid_loss_list = {'training time': training_time, 'loss':[], 'on time': True, 'num updates': trainer.get_num_updates()}
                for j in range(i+1):
                    print("validate on domain " + domains[j] )
                    for valid_sub_split in valid_subsets:
                        task.load_dataset(valid_sub_split, combine=False, domain=domains[j], epoch=0)
                    valid_losses, valid_ppl, valid_loss = validate(args, trainer, task, epoch_itr, valid_subsets, cur_domain=domains[j])
                    valid_loss_list['loss'].append(valid_loss)
                    if j < i:
                        args.ppl_increased.append(min(max(args.ppl_increased_min, valid_ppl - args.best_ppl[domains[j]]), args.ppl_increased_max) if domains[j] in args.best_ppl.keys() else args.ppl_increased_min)
                    args.best_ppl[domains[j]] = valid_ppl if domains[j] not in args.best_ppl.keys() else min(valid_ppl, args.best_ppl[domains[j]])
                args.valid_loss_list[name].append(valid_loss_list)
                print("best ppl for every domain: \n" + str(args.best_ppl))
                print("ppl increased: \n" + str(args.ppl_increased))
            else:
                valid_losses = [None]

            # only use first validation loss to update the learning rate
            lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

            # save checkpoint
            if epoch_itr.epoch % args.save_interval == 0:
                if args.cur_domain not in trainer.trained_domains:
                    trainer.trained_domains.append(args.cur_domain)
                checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

            reload_dataset = ':' in getattr(args, 'data', '')
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset, domain=domains[i])

            if training_time >= args.max_train_times[i]:
                break

        train_meter.stop()

        print('| done training for ' + args.cur_domain + ' in {:.1f} seconds'.format(train_meter.sum))

def random_pick(some_list,probabilities):
    x = random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            break
    return item

def train(args, trainer, task, epoch_itr, cur_domain=None, recover=False):
    """Train the model for one epoch."""
    # Update parameters every N batches
    assert cur_domain != None
    cur = args.domain_idx[cur_domain]
    domains = args.domains
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = (args.cur_max_update or math.inf) if not recover else args.recover_update
    if cur > 0 and len(args.ppl_increased) < cur: 
        print("reset ppl increased list")
        args.ppl_increased = [args.ppl_increased_min] * cur
    
    training_time_cur = 0
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        #print(i)
        num_updates = trainer.get_num_updates()
        
        for j in range(len(samples)):
            samples[j] = (cur_domain, samples[j])


        log_output = trainer.train_step(samples, cur=cur)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second and updates-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()
            trainer.get_meter('ups').reset()
        num_updates = trainer.get_num_updates()
        print(args.cur_domain, i, num_updates)

        time_threshold = args.last_validation_time + args.validation_interval_time
        #print(time_threshold)


        training_time = int(trainer.get_meter("train_wall").sum) if torch.distributed.get_rank() == 0 else 0
        print(training_time)
        training_time_tensor = torch.tensor(training_time, device='cuda')
        torch.distributed.broadcast(training_time_tensor, 0)
        training_time = training_time_tensor.item()
        training_time_cur = training_time
        
        
        if (
            not args.disable_validation
            and args.validation_interval_updates > 0
            and num_updates % args.validation_interval_updates == 0
            and num_updates > 0
        ):
            name = cur_domain + (' recover' if recover else '')
            if name not in args.valid_loss_list.keys():
                args.valid_loss_list[name] = []
            valid_loss_list = {'training time': training_time, 'loss':[], 'on time': False, 'num updates': trainer.get_num_updates()}

            args.ppl_increased = []
            for j in range(args.domain_idx[cur_domain] + 1):
                print("validate on domain " + args.domains[j])
                for valid_sub_split in valid_subsets:
                    task.load_dataset(valid_sub_split, combine=False, domain=args.domains[j], epoch=0)
                valid_losses, valid_ppl, valid_loss = validate(args, trainer, task, epoch_itr, valid_subsets, cur_domain=args.domains[j])

                valid_loss_list['loss'].append(valid_loss)

                if j < args.domain_idx[cur_domain]:
                    args.ppl_increased.append(min(max(args.ppl_increased_min, valid_ppl - args.best_ppl[domains[j]]), args.ppl_increased_max) if domains[j] in args.best_ppl.keys() else args.ppl_increased_min)
                args.best_ppl[domains[j]] = valid_ppl if domains[j] not in args.best_ppl.keys() else min(valid_ppl, args.best_ppl[domains[j]])
            
            print("best ppl for every domain: \n" + str(args.best_ppl))
            print("ppl increased: \n" + str(args.ppl_increased))
            args.valid_loss_list[name].append(valid_loss_list)
            if torch.distributed.get_rank() == 0:
                torch.save(args.valid_loss_list, os.path.join(args.save_dir, 'valid_loss.pt'))
            print(args.valid_loss_list)
        
        if (
            not args.disable_validation
            and args.validation_interval_time > 0
            and training_time >= time_threshold
        ):
            name = cur_domain + (' recover' if recover else '')
            if name not in args.valid_loss_list.keys():
                args.valid_loss_list[name] = []
            valid_loss_list = {'training time': training_time, 'loss':[], 'on time': True, 'num updates': trainer.get_num_updates()}

            for j in range(args.domain_idx[cur_domain] + 1):
                print("validate on domain " + args.domains[j])
                for valid_sub_split in valid_subsets:
                    task.load_dataset(valid_sub_split, combine=False, domain=args.domains[j], epoch=0)
                #print(name)
                valid_losses, valid_ppl, valid_loss = validate(args, trainer, task, epoch_itr, valid_subsets, cur_domain=args.domains[j])
                #print(valid_loss)
                valid_loss_list['loss'].append(valid_loss)

                args.best_ppl[domains[j]] = valid_ppl if domains[j] not in args.best_ppl.keys() else min(valid_ppl, args.best_ppl[domains[j]])
            print("best ppl for every domain: \n" + str(args.best_ppl))



            args.last_validation_time = training_time
            
            args.valid_loss_list[name].append(valid_loss_list)
            if torch.distributed.get_rank() == 0:
                torch.save(args.valid_loss_list, os.path.join(args.save_dir, 'valid_loss.pt'))
            print(args.valid_loss_list)
            

        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            if args.cur_domain not in trainer.trained_domains:
                trainer.trained_domains.append(args.cur_domain)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break
            
        if training_time >= args.max_train_times[cur]:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    return training_time_cur

def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets, cur_domain=None):
    """Evaluate the model on the validation set(s) and return the losses."""
    
    assert cur_domain != None
    cur = args.domain_idx[cur_domain]

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())
        i = 0
        for sample in progress:
            i += 1
            #print(i, len(progress))
            log_output, sample_size = trainer.valid_step(cur_domain, sample, cur=cur)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses, stats['ppl'], stats['loss'].avg


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
