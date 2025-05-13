from __future__ import print_function
from __future__ import division

import argparse
import random
import torch.optim as optim
import torch.utils.data
import os
from tqdm import tqdm

import dcgan
import utils as myutils


if __name__ == '__main__':

    FileNotFoundError = IOError

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='mpond')
    # parser.add_argument('--dataset', default='fluvial2020')
    parser.add_argument('--dataset', default='grdecl')
    parser.add_argument('--dataroot', default='mpond', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=6, help='input image channels')
    parser.add_argument('--nz', type=int, default=60, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=15000, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', default=False, action='store_true', help='use batchnorm or not')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', default=False, action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--maxSamples', type=int, default=100000, help='Max samples to use for training')
    parser.add_argument('--maxFiles', type=int, default=50, help='Max files to use for training')
    parser.add_argument('--strideX', type=int, default=11, help='strid in x direction')
    parser.add_argument('--strideY', type=int, default=7, help='strid in y direction')
    parser.add_argument('--device', default='cuda:0', help='device to run')

    opt = parser.parse_args()
    print(opt)

    if opt.experiment is None:
        opt.experiment = 'grdecl'
    opt.experiment += "_" + str(opt.strideX)
    opt.experiment += "_" + str(opt.strideY)
    opt.experiment += "_" + str(opt.maxFiles)
    opt.experiment += "_" + str(opt.nz)

    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = 1234  # random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    try:
        print('===========================')
        # we have two GPUs
        print(torch.cuda.device_count())
        print("device 0 ", torch.cuda.get_device_name(0))
        print("device 1 ", torch.cuda.get_device_name(1))
    except Exception as e:
        pass

    # device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # print("device type", device.type)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'


    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers).to(device)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers).to(device)
    netG.apply(myutils.weights_init)

    # netg_prev = r'C:\NORCE_Projects\DISTINGUISH\code\src\gan-geosteering-prestudy-internal-master\grdecl_32_32_50_60\netG_epoch_777.pth'
    # netG.load_state_dict(torch.load(netg_prev))

    if opt.netG != '':  # load checkpoint if needed
        load_results = netG.load_state_dict(torch.load(opt.netG))
        print("Loaded weights")
        print(load_results)

    print(netG)

    if opt.noBN:
        netD = dcgan.DCGAN_D_nobn(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers).to(device)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers).to(device)

    netD.apply(myutils.weights_init)
    # netd_prev = r'C:\NORCE_Projects\DISTINGUISH\code\src\gan-geosteering-prestudy-internal-master\grdecl_32_32_50_60\netD_epoch_777.pth'
    # netD.load_state_dict(torch.load(netd_prev))

    if opt.netD != '':
        load_result = netD.load_state_dict(torch.load(opt.netD))
        print("Loaded weights")
        print(load_result)
    print(netD)

    if opt.dataset == 'mpond':
        dataset = \
            myutils.CustomDatasetFromMAT(1888, opt.imageSize, opt.imageSize, transforms=None)
        assert dataset
    elif opt.dataset == 'fluvial1':
        dataset = \
            myutils.CustomDatasetFromRMS("fluvial", opt.imageSize, opt.imageSize, transforms=None,
                                         channels=nc)
    elif opt.dataset == 'fluvial2020':
        # dataset = \
        #     myutils.CustomDatasetFromRMS("grids2020vari-porocity", opt.imageSize, opt.imageSize, transforms=None,
        #                                  channels=nc,
        #                                  constant_axis=1,
        #                                  min_facies_id=0,
        #                                  do_flip=False)
        dataset = \
            myutils.CustomDatasetFromRMS("grids10-2020-vari-porosity", opt.imageSize, opt.imageSize, transforms=None,
                                         channels=nc,
                                         constant_axis=1,
                                         do_flip=False,
                                         porous=6,
                                         max_files=opt.maxFiles,
                                         max_samples=opt.maxSamples,
                                         stride_x=opt.strideX,
                                         stride_y=opt.strideY)

    # TODO make a new converter that adds porocity
    # TODO note there is a function that truncates "pixels" which needs to be removed
    elif opt.dataset == 'grdecl':
        # dataset = \
        #     myutils.CustomDatasetFromRMS("grids2020vari-porocity", opt.imageSize, opt.imageSize, transforms=None,
        #                                  channels=nc,
        #                                  constant_axis=1,
        #                                  min_facies_id=0,
        #                                  do_flip=False)
        folder_name = '../../../data/gan_training'
        dataset = \
            myutils.CustomDatasetFromGRD(folder_name, opt.imageSize, opt.imageSize, transforms=None,
                                         channels=nc,
                                         constant_axis=1,
                                         do_flip=False,
                                         porous=6,
                                         max_files=opt.maxFiles,
                                         max_samples=opt.maxSamples,
                                         stride_x=opt.strideX,
                                         stride_y=opt.strideY)

    else:
        raise Exception('datatype not defined')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    noise_vec = torch.FloatTensor(opt.batchSize, nz, 1, 1).to(device)
    fixed_noise_vec = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1).to(device)
    #fixed_zero_vec = torch.FloatTensor(opt.batchSize, nz, 1, 1).zero_().to(device)
    fixed_zero_vec = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1e-2).to(device)

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999), amsgrad=True)
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), amsgrad=True)
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    gen_iterations = 0
    for epoch in tqdm(range(opt.niter)):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if (opt.netD == '' and gen_iterations < 25) or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters

            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                try:
                    data = next(data_iter)
                except Exception:
                    data_iter = iter(dataloader)
                    data = next(data_iter)

                i += 1

                netD.zero_grad()

                # train with real
                x_real = data[0].to(device)
                errD_real = netD(x_real).mean()


                # train with fake
                x_fake = netG(noise_vec.normal_(0, 1)).detach()
                errD_fake = netD(x_fake).mean()

                errD = -(errD_real - errD_fake)

                #errD.baackword computes the gradients with respect to the original parameters
                errD.backward()
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation

            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            x_fake = netG(noise_vec)
            errG = -1 * netD(x_fake).mean()

            errG.backward()
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f' %
                  (epoch, opt.niter-1, i, len(dataloader), gen_iterations,
                   errD.item(), errG.item(), errD_real.item(), errD_fake.item()))
            if gen_iterations % 100 == 0:
                myutils.save_image(x_real,
                                   '{0}/real_samples_{1}.png'.format(opt.experiment, gen_iterations % 5000),
                                   device)
                x_fake = netG(fixed_noise_vec)
                myutils.save_image(x_fake,
                                   '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations % 5000),
                                   device)
                x_fake_at_zero = netG(fixed_zero_vec)
                myutils.save_image(x_fake_at_zero,
                                   '{0}/fake_sample0_{1}.png'.format(opt.experiment, gen_iterations % 5000),
                                   device)

        if (epoch+1) % 100 == 0:
            # do checkpointing
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch+1))
            torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch+1))

            print("Saved after epoch {}".format(epoch+1))

            # saving epochal images
            myutils.save_image(x_real,
                               '{0}/real_epoch_{1}.png'.format(opt.experiment, epoch+1),
                               device)
            x_fake = netG(fixed_noise_vec)
            myutils.save_image(x_fake,
                               '{0}/fake_epoch_{1}.png'.format(opt.experiment, epoch+1),
                               device)
            x_fake_at_zero = netG(fixed_zero_vec)
            myutils.save_image(x_fake_at_zero,
                               '{0}/fake_epoch_at0_{1}.png'.format(opt.experiment, epoch+1),
                               device)
