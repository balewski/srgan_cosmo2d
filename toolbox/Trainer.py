import os,time
from pprint import pprint,pformat
import socket  # for hostname
import numpy as np
import torch

import logging  # to screen

from toolbox.Dataloader_H5 import get_data_loader  # 1-cell (code is simpler)
from toolbox.Util_IOfunc import read_yaml
#from toolbox.Model_3D import G,D

from toolbox.Model_2d import Generator, Discriminator, ContentLoss

#from toolbox.tmp_figures import ( plt_slices, plt_power )
from toolbox.tmp_TBSwriter import TBSwriter

import torch.optim as optim

#............................
#............................
#............................
class Trainer(TBSwriter):
#...!...!..................
    def __init__(self, params):
        
        assert torch.cuda.is_available() 
        self.params = params
        self.verb=params['verb']
        self.isRank0=params['world_rank']==0
        self.device = torch.cuda.current_device()        

        if params['world_rank']<20:
            logging.info('T:ini world rank %d of %d, local rank %d, host=%s  see device=%d'%(params['world_rank'],params['world_size'],params['local_rank'],socket.gethostname(),self.device))

        expDir=params['out_path']
        if self.isRank0:
            TBSwriter.__init__(self,expDir)
            expDir2=os.path.join(expDir, 'checkpoints')
            if not os.path.isdir(expDir2):  os.makedirs(expDir2)
            params['checkpoint_path'] = expDir2
        #params['resuming'] =  params['resume_checkpoint'] and os.path.isfile(params['checkpoint_path'])
        
        optTorch=params['opt_pytorch']
        # EXTRA: enable cuDNN autotuning.
        torch.backends.cudnn.benchmark = optTorch['autotune']
        torch.autograd.set_detect_anomaly(optTorch['detect_anomaly'])

        if self.verb>1: logging.info('T:params %s'%pformat(params))

        # ...... END OF CONFIGURATION .........    
        if self.verb:
          logging.info('T:imported PyTorch ver:%s verb=%d'%(torch.__version__,self.verb))
          logging.info('T:rank %d of %d, prime data loaders'%(params['world_rank'],params['world_size']))


        params['shuffle']=True  
        self.train_loader = get_data_loader(params, 'train',verb=self.verb)
        params['shuffle']=True # use False for reproducibility
        self.valid_loader = get_data_loader(params, 'valid', verb=self.verb)
        
        inpMD=self.train_loader.dataset.conf
        if self.verb:
          logging.info('T:rank %d of %d, data loaders initialized'%(params['world_rank'],params['world_size']))
          logging.info('T:train-data: %d steps, localBS=%d, globalBS=%d'%(len(self.train_loader),self.train_loader.batch_size,params['global_batch_size']))
          
          logging.info('T:valid-data: %d steps'%(len(self.valid_loader)))        
          logging.info('T:meta-data from h5: %s'%pformat(inpMD))
          
        if params['world_rank']==0 and 0:  # for now needed for cub-shapes
            someLoader=self.train_loader
            print('TI:dataset example,train loader len=numsteps:',len(someLoader))
            dataD=next(iter(someLoader))

            print('TI:one sample=',list(dataD))
            for x in ['input', 'target']:
                y=dataD[x]
                print(x,type(y),y.shape)

            # needed for Summary()
            seen_inp_shape=dataD['input'].shape[1:] # skip batch dimension
            tgt_shape=dataD['target'].shape[1:]

            
        if params['world_size']>1:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel
            assert dist.is_initialized()
            dist.barrier()
            self.dist=dist # needed by this class member methods
            # wait for all ranks to finish downloading the data - lets keep some order
            
        if self.verb:
            # must know the number of steps to decided how often to print
            params['text_log_interval_steps']=max(1,len(self.train_loader)//params['text_log_freq_per_epoch'])
            #logging.info('T:logger iterD: %s'%str(self.iterD))
                         
        if self.verb:
            logging.info('T:assemble G+D models')

        #self.model=MyModel(params['model'], verb=self.verb).to(self.device)
        #m2mCf=params['map2map_conf']

        #inp_chan=inpMD['lowResShape'][0]
        #out_chan=inpMD['highResShape'][0]
        #scale_fact=m2mCf['scale_factor']
        
        #self.G_model = G(inp_chan,out_chan, scale_factor=scale_fact,verb=self.verb)
        self.G_model = Generator(params['num_inp_chan'],verb=self.verb)
        self.D_model = Discriminator(params['num_inp_chan'],verb=self.verb)

        # add models to TB - takes 1 min
        if self.isRank0:
            gr2tb=params['model_conf']['tb_add_graph']
            dataD=next(iter(self.train_loader))            
            if 'G'==gr2tb:
                t1=time.time()
                self.TBSwriter.add_graph(self.G_model,dataD['input'].to('cpu'))
                t2=time.time()
                print('TB G-graph done elaT=%.1f'%(t2-t1))
            if 'D'==gr2tb:
                t1=time.time()
                self.TBSwriter.add_graph(self.D_model,dataD['target'].to('cpu'))
                t2=time.time()
                print('TB D-graph done elaT=%.1f'%(t2-t1),gr2tb)

        self.G_model=self.G_model.to(self.device)
        self.D_model=self.D_model.to(self.device)
        
        trCf=params['train_conf']
        self.PG_opt = optim.Adam(self.G_model.parameters(), lr=trCf['PG_LR']['init'])
        self.G_opt = optim.Adam(self.G_model.parameters(), lr=trCf['G_LR']['init'])
        self.D_opt = optim.Adam(self.D_model.parameters(), lr=trCf['D_LR']['init'])


        self.G_sched = torch.optim.lr_scheduler.StepLR(self.G_opt, trCf['G_LR']['decay_epochs'], gamma=trCf['G_LR']['gamma'], verbose=0)
        self.D_sched = torch.optim.lr_scheduler.StepLR(self.D_opt, trCf['D_LR']['decay_epochs'], gamma=trCf['D_LR']['gamma'], verbose=0)


        # Loss function.
        self.psnr_criterion   = torch.nn.MSELoss().to(self.device)     # PSNR metrics.
        self.pixel_criterion  = torch.nn.MSELoss().to(self.device)     # Pixel loss.
        self.content_criterion= ContentLoss().to(self.device)    # Content loss is activation(?) of 36th layer in VGG19
        #  Binary Cross Entropy between the target and the input probabilities
        self.adversarial_criterion = torch.nn.BCELoss().to(self.device)     # Adversarial loss.
 
        if trCf['resume']:
            print("Resuming...")
            never_tested1 # must sync weights on all ranks
            Ta=time.time()
            if resume_p_weight != "":
                generator.load_state_dict(torch.load(resume_p_weight))
            else:
                discriminator.load_state_dict(torch.load(resume_d_weight))
                generator.load_state_dict(torch.load(resume_g_weight))
            Tb=time.time()
            print('M:model loaded from disc, elaT=%.1f sec'%(Tb-Ta))
        
        if self.verb:
            logging.info('T: D+G models created')#, seen inp_shape: %s',str(seen_inp_shape))
            logging.info(self.G_model.summary())
            logging.info(self.D_model.summary())

            [Gpr,Dpr]=params['model_conf']['print_summary']
            if Dpr+Gpr>0: from torchsummary import summary  # not visible if loaded earlier

            if Gpr & 1:  logging.info('T:generator layers %s'%pformat(self.G_model))
            if Gpr & 2:  logging.info('T:generator summary %s'%pformat(summary(self.G_model,tuple(seen_inp_shape))))
            if Dpr & 1:  logging.info('T:discriminator layers %s'%pformat(self.D_model))
            if Dpr & 2:  logging.info('T:discriminator summary %s'%pformat(summary(self.D_model,tuple(tgt_shape))))
            
        if params['world_size']>1:
            self.G_model = DistributedDataParallel(self.G_model,
                              device_ids=[params['local_rank']],output_device=[params['local_rank']])
            self.D_model = DistributedDataParallel(self.D_model,
                              device_ids=[params['local_rank']],output_device=[params['local_rank']])
        # note, using DDP assures the same as average_gradients(self.model), no need to do it manually
        

        self.startEpoch = 0
        self.epoch = self.startEpoch

        if self.isRank0:  # create summary record
            tot_train_samp= params['world_size']*len(self.train_loader)
            self.sumRec={'train_params':params,
                         'hostName' : socket.gethostname(),
                         'numRanks': params['world_size'],
                         'state': 'model_build',
                         'input_meta':inpMD,
                         'total_train_samp':tot_train_samp,
                         'trainTime_sec':-1,
                         'loss_valid':-1,
                         'pytorch': str(torch.__version__),
                         'epoch_start': int(self.startEpoch),
            }

      
#...!...!..................
    def train(self):
        if self.verb:
            logging.info("Starting Training Loop..., myRank=%d "%(self.params['world_rank']))
        
        startTrain = time.time()
        TperEpoch=[]        
        trCf=self.params['train_conf']
        T0=time.time()
 
        start_p_epoch=trCf['start_p_epoch']
        p_epochs=trCf['p_epochs']
        start_epoch=trCf['start_epoch']
        epochs=trCf['epochs']

        print('M:train start, epochs range: [%d,%d]'%(start_p_epoch, p_epochs),'device=',self.device)

        # Initialize the evaluation indicators for the training stage of the generator model.
        best_psnr_value = 0.0
        # Train the generative network stage.
        for epoch in range(start_p_epoch, p_epochs):
            Ta=time.time()
            # Train each epoch for generator network.
            self.train_generator( epoch)
            Tb=time.time()
            # Verify each epoch for generator network.
            psnr_value = self.validate( epoch, "generator")
            Tc=time.time()
            # Determine whether the performance of the generator network under epoch is the best.
            is_best = psnr_value > best_psnr_value 
            best_psnr_value = max(psnr_value, best_psnr_value)

            if self.isRank0:
                # Save the weight of the generator network under epoch. If the performance of the generator network under epoch is best, save a file ending with `-best.pth` in the `results` directory.
                exp_dir2=self.params['checkpoint_path']
                torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, f"p_epoch{epoch + 1}.pth"))
                if is_best:
                    torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "p-best.pth"))

            print('PT:epoch %d  GEN train %.1f sec, val=%.1f sec, psnr=%.1f best_psnr=%.1f, elaT=%.1f min\n'%(epoch+1,Tb-Ta,Tc-Tb,psnr_value,best_psnr_value,(Tc-T0)/60))
        # Save the weight of the last generator network under Epoch in this stage.
        torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "p-last.pth"))
        # Initialize the evaluation index of the adversarial network training phase.
        best_psnr_value = 0.0
        psnrA=0.; psnrB=0.; psnrC=0. # for best epoch tag
        # Load the model weights with the best indicators in the previous round of training.
        self.G_model.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))
        exp_dir2=self.params['checkpoint_path']
        gdict={}#tmp
        
        # Training the adversarial network stage.
        for epoch in range(start_epoch, epochs):
            
            # Apply learning rate warmup
            if epoch < trCf['adv_warmup_epochs']:
                self.G_opt.param_groups[0]['lr']=trCf['G_LR']['init']*float(epoch+1.)/trCf['adv_warmup_epochs']
                self.D_opt.param_groups[0]['lr']=trCf['D_LR']['init']*float(epoch+1.)/trCf['adv_warmup_epochs']       
                
                
            # Train each epoch for adversarial network.
            Ta=time.time()
            self.train_adversarial(epoch,gdict)
            Tb=time.time()
            # Verify each epoch for adversarial network.
            psnr_value = self.validate( epoch, "adversarial")
            Tc=time.time()
            #  use running averag over last 3 epochs to tag best epoch
            is_best=False
            if  epoch>0.3* epochs:
                psnrC=psnrB
                psnrB=psnrA
                psnrA=psnr_value
                psnrAvr=(psnrA+psnrB+psnrC)/3.
                if best_psnr_value < psnrAvr:
                    is_best =True
                    best_psnr_value = psnrAvr

            # Save the weight of the adversarial network under epoch. If the performance of the adversarial network
            # under epoch is the best, it will save two additional files ending with `-best.pth` in the `results` directory.
            if self.isRank0:
                torch.save(self.D_model.state_dict(), os.path.join(exp_dir2, f"d-epoch{epoch + 1}.pth"))
                torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, f"g-epoch{epoch + 1}.pth"))
            if is_best and self.isRank0:
                torch.save(self.D_model.state_dict(), os.path.join(exp_dir2, "d-best%d.pth"%(epoch+1)))
                torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "g-best%d.pth"%(epoch+1)))
                txt='Epoch %d ADV best PSNR=%.2f elaT=%.1f sec'%(epoch+1,best_psnr_value,time.time()-T0)
                self.TBSwriter.add_text('summary',txt , epoch+1)
                print(txt)
            print('M:epoch %d  ADV train %.1f sec, val=%.1f sec, psnr=%.2f best_psnr=%.2f, elaT=%.1f min\n'%(epoch+1,Tb-Ta,Tc-Tb,psnr_value,best_psnr_value,(Tc-T0)/60))
            # Adjust the learning rate of the adversarial model.
            self.D_sched.step()
            self.G_sched.step()
            
            if self.isRank0: # monitor LR
                recLR={'G':self.G_opt.param_groups[0]['lr'],'D':self.D_opt.param_groups[0]['lr']}
                #print('JJ:',epoch, recLR)
                self.TBSwriter.add_scalars('ADV_LR',recLR , epoch+1)
                

        if self.isRank0:
            # Save the weight of the adversarial network under the last epoch in this stage.

            torch.save(self.D_model.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
            torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "g-last.pth"))
        print('T:done exp_dir2:',exp_dir2)


        if self.isRank0:  # create summary record
          # add info to summary
          if 1:
            rec={'epoch_stop':epoch, 'state':'model_trained'} 
            rec['trainTime_sec']=time.time()-startTrain
            #1 rec['timePerEpoch_sec']=[float('%.2f'%x) for x in [tAvr,tStd] ]
            self.sumRec.update(rec)
            return
 
#...!...!..................
    def train_generator(self, epoch) -> None:
        """Training the generator network.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
            epoch (int): number of training cycles.
        """
        # Calculate how many iterations there are under epoch.
        batches = len(self.train_loader)
        # Set generator network in training mode.
        self.G_model.train()

        for index, (lr, hr) in enumerate(self.train_loader):
            # Copy the data to the specified device.
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            #print('tt',lr.shape,hr.shape,hr.dtype)
            # Initialize the gradient of the generator model.
            if self.params['opt_pytorch']['zerograd']:
                # EXTRA: Use set_to_none option to avoid slow memsets to zero
                self.G_model.zero_grad(set_to_none=True)
            else:
                self.G_model.zero_grad()
            # Generate super-resolution images.
            sr = self.G_model(lr)
            #print('tt2',sr.shape,sr.dtype); ok67
            # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
            pixel_loss = self.pixel_criterion(sr, hr)
            # Update the weights of the generator model.
            pixel_loss.backward()
            self.PG_opt.step()
            # Write the loss during training into Tensorboard.
            iters = index + epoch * batches + 1
            if self.isRank0: self.TBSwriter.add_scalar("Train_Generator/Loss_pixel", pixel_loss.item(), iters)
            # Print the loss function every ten iterations and the last iteration in this epoch.
            if (index + 1) % 30 == 0 or (index + 1) == batches:
                print(f"Train Epoch[{epoch + 1:04d}]({index + 1:05d}/{batches:05d}) "
                      f"Loss: {pixel_loss.item():.6f}.")


    def train_adversarial(self, epoch, gdict) -> None:
        """Training the adversarial network.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The loader of the training dataset.
            epoch (int): number of training cycles.
        """
        trCf=self.params['train_conf']
        # Calculate how many iterations there are under Epoch.
        batches = len(self.train_loader)
        # Set adversarial network in training mode.
        self.D_model.train()
        self.G_model.train()

        for index, (lr, hr) in enumerate(self.train_loader):
            # Copy the data to the specified device.
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            label_size = lr.size(0)
            # Create label. Set the real sample label to 1, and the false sample label to 0.
            real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=self.device)
            fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=self.device)

            # Initialize the gradient of the discriminator model.
            if self.params['opt_pytorch']['zerograd']:
                self.D_model.zero_grad(set_to_none=True)
            else:
                self.D_model.zero_grad()
            # Calculate the loss of the discriminator model on the high-resolution image.
            output = self.D_model(hr)
            d_loss_hr = self.adversarial_criterion(output, real_label)
            d_loss_hr.backward()
            d_hr = output.mean().item()
            # Generate super-resolution images.
            sr = self.G_model(lr)
            # Calculate the loss of the discriminator model on the super-resolution image.
            output = self.D_model(sr.detach())
            d_loss_sr = self.adversarial_criterion(output, fake_label)
            d_loss_sr.backward()
            d_sr1 = output.mean().item()
            # Update the weights of the discriminator model.
            d_loss = d_loss_hr + d_loss_sr
            self.D_opt.step()

            # Initialize the gradient of the generator model.
            if self.params['opt_pytorch']['zerograd']:
                self.G_model.zero_grad(set_to_none=True)
            else: 
                self.G_model.zero_grad()
            # Calculate the loss of the discriminator model on the super-resolution image.
            output = self.D_model(sr) #  len=BS filled w/ probabilities of being real

            # Perceptual_loss= weighted sum:  pixel + content +  adversarial + power_spect
            adversarial_loss =  self.adversarial_criterion(output, real_label)
            pixel_loss =  self.pixel_criterion(sr, hr.detach())
            perceptual_loss =  self.content_criterion(sr, hr.detach())

            ''' add later
            # Add spectral loss
            mean,sdev=f_torch_image_spectrum(f_invtransform(sr),1,gdict['spec_r'].to(self.device),gdict['spec_ind'].to(self.device))
            spec_lmean,spec_lsdev=loss_spectrum(mean,gdict['mean_spec_val'].to(self.device),sdev,gdict['sdev_spec_val'].to(self.device),image_size)
            spect_loss= spect_lavr_weight*spec_lmean+ spect_lstd_weight*spec_lsdev
            '''
            
            # Update the weights of the generator model.
            g_loss = trCf['pixel_weight'] *pixel_loss + trCf['content_weight'] *perceptual_loss + trCf['adversarial_weight'] *adversarial_loss #1+ spect_loss
            g_loss.backward()
            self.G_opt.step()
            d_sr2 = output.mean().item()
            #print('J:d_sr1,2',d_sr1,d_sr2)
            # Write the loss during training into Tensorboard.
            iters = index + epoch * batches + 1
            if self.isRank0:
                self.TBSwriter.add_scalar("Train_Adversarial/D_Loss", d_loss.item(), iters)
                self.TBSwriter.add_scalar("Train_Adversarial/G_Loss", g_loss.item(), iters)
                self.TBSwriter.add_scalar("Train_Adversarial/G_pixel", pixel_loss.item(), iters)
                self.TBSwriter.add_scalar("Train_Adversarial/G_content", perceptual_loss.item(), iters)
                '''
                recSPE={'2*lmean':2*spec_lmean,'lstd':spec_lsdev}
                writer.add_scalars("Train_Adversarial/Power_spect", recSPE, iters)

                writer.add_scalar("Train_Adversarial/G_spect", spect_loss, iters)
                #writer.add_scalar("Train_Adversarial/Power_lsdev", spec_lsdev, iters)
                #writer.add_scalar("Train_Adversarial/Power_lmean2", aa, iters)
                '''
                self.TBSwriter.add_scalar("Train_Adversarial/G_xentr", adversarial_loss.item(), iters)
                self.TBSwriter.add_scalar("Train_Adversarial/D_hr", d_hr, iters)
                self.TBSwriter.add_scalar("Train_Adversarial/D_sr1", d_sr1, iters)
                #writer.add_scalar("Train_Adversarial/D_SR2", d_sr2, iters)
                # Print the loss function every ten iterations and the last iteration in this Epoch.
            if index % self.params['text_log_interval_steps']==0 and self.verb:
                print(f"Train stage: adversarial "
                      f"Epoch[{epoch + 1:04d}]({index + 1:05d}/{batches:05d}) "
                      f"D Loss: {d_loss.item():.3g} G Loss: {g_loss.item():.6f} "
                      f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.3g}/{d_sr2:.3g}.")


    def validate(self, epoch, stage) -> float:
        """Verify the generator model.

        Args:
            valid_dataloader (torch.utils.data.DataLoader): loader for validating dataset.
            epoch (int): number of training cycles.
            stage (str): In which stage to verify, one is `generator`, the other is `adversarial`.

        Returns:
            PSNR value(float).
        """
        # Calculate how many iterations there are under epoch.
        batches = len(self.valid_loader)
        # Set generator model in verification mode.
        self.G_model.eval()
        # Initialize the evaluation index.
        total_psnr_value = 0.0

        with torch.no_grad():
            for index, (lr, hr) in enumerate(self.valid_loader):
                # Copy the data to the specified device.
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                # Generate super-resolution images.
                sr = self.G_model(lr)
                # Calculate the PSNR indicator.
                mse_loss = self.psnr_criterion(sr, hr)
                psnr_value = 10 * torch.log10(1 / mse_loss).item()
                total_psnr_value += psnr_value

            avg_psnr_value = total_psnr_value / batches
            # Write the value of each round of verification indicators into Tensorboard.
            if self.isRank0:
                if stage == "generator":
                    self.TBSwriter.add_scalar("Val_Generator/PSNR", avg_psnr_value, epoch + 1)
                elif stage == "adversarial":
                    self.TBSwriter.add_scalar("Val_Adversarial/PSNR", avg_psnr_value, epoch + 1)
                    
        return avg_psnr_value
