import os,time
from pprint import pprint,pformat
import socket  # for hostname
import numpy as np
import torch

import logging  # to screen

from toolbox.Dataloader_H5 import get_data_loader  # 1-cell (code is simpler)
from toolbox.Util_IOfunc import read_yaml,dateT2Str
from toolbox.Util_H5io3 import  write3_data_hdf5
from toolbox.Model_2d import Generator, Discriminator, ContentLoss
from toolbox.Util_Torch import all_reduce_dict, compute_fft, transf_img2field_torch, torchD_to_floatD
#from toolbox.tmp_figures import ( plt_slices, plt_power )
from toolbox.tmp_TBSwriter import TBSwriter
from toolbox.RingAverageCheck import RingAverageCheck

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

        expDir=params['exp_path']
        if self.isRank0:
            TBSwriter.__init__(self,expDir)
            expDir2=os.path.join(expDir, 'checkpoints')
            if not os.path.isdir(expDir2):  os.makedirs(expDir2)
            params['checkpoint_path'] = expDir2
            expDir3=os.path.join(expDir, 'snapshots')
            if not os.path.isdir(expDir3):  os.makedirs(expDir3)
            params['val_mon_path'] = expDir3
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
          self.add_tbsummary_record(pformat(inpMD))
          
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
        else:
            self.dist=None # single GPU training
            
        if self.verb:
            # must know the number of steps to decided how often to print
            params['text_log_interval_steps']=max(1,len(self.train_loader)//params['text_log_interval/epochs'])
            #logging.info('T:logger iterD: %s'%str(self.iterD))
                         
        if self.verb:
            logging.info('T:assemble G+D models')

        self.G_model = Generator(params['num_inp_chan'],verb=self.verb)
        self.D_model = Discriminator(params['num_inp_chan'],params['model_conf']['D'],verb=self.verb)

        # add models to TB - takes 1 min
        if self.isRank0:
            gr2tb=params['model_conf']['tb_model_graph']
            (lrB, hrB)=next(iter(self.train_loader))            
            if 'G'==gr2tb:
                t1=time.time()
                self.TBSwriter.add_graph(self.G_model,lrB.to('cpu'))
                t2=time.time()
                print('TB G-graph done elaT=%.1f'%(t2-t1))
            if 'D'==gr2tb:
                t1=time.time()
                self.TBSwriter.add_graph(self.D_model,hrB.to('cpu'))
                t2=time.time()
                print('TB D-graph done elaT=%.1f'%(t2-t1),gr2tb)

        self.G_model=self.G_model.to(self.device)
        self.D_model=self.D_model.to(self.device)
        
        trCf=params['train_conf']
        self.PG_opt = optim.Adam(self.G_model.parameters(), lr=trCf['PG_LR']['init'])
        self.G_opt = optim.Adam(self.G_model.parameters(), lr=trCf['G_LR']['init'])
        self.D_opt = optim.Adam(self.D_model.parameters(), lr=trCf['D_LR']['init'])


        ''' disable schedulers
        self.G_sched = torch.optim.lr_scheduler.StepLR(self.G_opt, trCf['G_LR']['decay/epochs'], gamma=trCf['G_LR']['gamma'], verbose=0)
        self.D_sched = torch.optim.lr_scheduler.StepLR(self.D_opt, trCf['D_LR']['decay/epochs'], gamma=trCf['D_LR']['gamma'], verbose=0)
        '''


        #.... Loss functions.
        self.psnr_criterion   = torch.nn.MSELoss().to(self.device)     # PSNR metrics.
        self.pixel_criterion  = torch.nn.MSELoss().to(self.device)     # Pixel loss.
        self.fft_criterion   = torch.nn.MSELoss().to(self.device)     # FFT loss.
        self.msum_criterion  = torch.nn.MSELoss().to(self.device)     # mass_sum loss.
        
        self.content_criterion= ContentLoss().to(self.device)    # Content loss is activation(?) of 36th layer in VGG19
        #  Binary Cross Entropy between the target and the input probabilities
        self.adversarial_criterion = torch.nn.BCELoss().to(self.device)     # Adversarial loss.            

        # initialize early-stop due to stuck discr(G) for too many epochs
        esCf=trCf['early_stop_discr']
        self.earlyStopRing=RingAverageCheck(
            func= lambda avr,std: avr+std < esCf['discr_G_thres'],
            numCell=esCf['ring_size/epochs'],initVal=0.2)  # init at large values to not trip during filling of the ring
        
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
            logging.info(self.G_model.short_summary())
            logging.info(self.D_model.short_summary())
            self.add_tbsummary_record(str(self.G_model.short_summary()))
            self.add_tbsummary_record(str(self.D_model.short_summary()))


            Gpr=params['model_conf']['G']['print_summary']
            Dpr=params['model_conf']['D']['print_summary']
        
            
            if Dpr+Gpr>0: from torchsummary import summary  # not visible if loaded earlier

            if Gpr & 1:  logging.info('T:generator layers %s'%pformat(self.G_model))
            if Gpr & 2:  logging.info('T:generator summary %s'%pformat(summary(self.G_model,tuple(params['lr_img_shape']))))
            if Dpr & 1:  logging.info('T:discriminator layers %s'%pformat(self.D_model))
            if Dpr & 2:  logging.info('T:discriminator summary %s'%pformat(summary(self.D_model,tuple(params['hr_img_shape']))))
            
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
                         'total_train_samp':tot_train_samp,
                         'train_duration/sec':-1,
                         'train_date': dateT2Str(time.localtime()),
                         'loss_valid':-1,
                         'pytorch': str(torch.__version__),
                         'epoch_start': int(self.startEpoch),
            }

      
#...!...!..................
    def train(self):
        if self.verb:
            logging.info("Starting Training Loop..., myRank=%d "%(self.params['world_rank']))
            
        
        T0 = time.time()
        trCf=self.params['train_conf']
         
        start_pre_epoch=trCf['start_pre_epoch']
        pre_epochs=trCf['pre_epochs']
        start_epoch=trCf['start_epoch']
        epochs=trCf['epochs']
        
        if self.verb:
            txt='exp=%s  pretrain, epochs [%d,%d], numGpu=%d, date=%s' %(self.params['exp_name'],start_pre_epoch, pre_epochs,self.params['world_size'],self.sumRec['train_date'])
            self.add_tbsummary_record(txt)
            logging.info(txt)
            
                                         
        # Initialize the evaluation indicators for the training stage of the generator model.
        best_psnr_value = 0.0
        # Train the generative network stage.
        for epoch in range(start_pre_epoch, pre_epochs):
            Ta=time.time()
            # Train each epoch for generator network.
            self.pretrain_generator( epoch)
            Tb=time.time()
            # Verify each epoch for generator network.
            psnr_value = self.validate( epoch, "gen")
            Tc=time.time()
            # Determine whether the performance of the generator network under epoch is the best.
            is_best = psnr_value > best_psnr_value 
            best_psnr_value = max(psnr_value, best_psnr_value)

            if self.isRank0:
                # Save the weight of the generator network under epoch. If the performance of the generator network under epoch is best, save a file ending with `-best.pth` in the `results` directory.
                exp_dir2=self.params['checkpoint_path']
                if epoch% self.params['checkpoint_interval/epochs']==0:
                    torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, f"pre_epoch{epoch + 1}.pth"))
                if is_best:
                    torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "p-best.pth"))

            if self.verb: logging.info('PreTrain: epoch %d  train=%.1f sec, val=%.1f sec, psnr=%.1f best_psnr=%.1f, elaT=%.1f min\n'%(epoch,Tb-Ta,Tc-Tb,psnr_value,best_psnr_value,(Tc -T0)/60))
        if self.isRank0:
            # Save the weight of the last generator network under Epoch in this stage.
            torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "p-last.pth"))
        # Initialize the evaluation index of the adversarial network training phase.
        best_psnr_value = 0.0; best_epoch=-1; best_cnt={}
        psnrA=0.; psnrB=0.; psnrC=0. # for best epoch tag
        
        if 0:  # I'm not sure if the last model must be the best, OFF for now, skips IO
            # Load the model weights with the best indicators in the previous round of training.
            self.G_model.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))
       
        if self.verb:
            txt='Start adv_train, epochs [%d,%d],  numGpu=%d, elapsedTime=%.1f min' %(start_epoch, epochs,self.params['world_size'],(time.time()-T0)/60.)
            #self.TBSwriter.add_text('summary',txt , global_step=1)
            self.add_tbsummary_record(txt)
            logging.info(txt)
            

        if self.isRank0:
            TperEpoch=[]
            locTrainSamp=len(self.train_loader)*self.train_loader.batch_size
            locValSamp=len(self.valid_loader)*self.valid_loader.batch_size
            kfac=self.params['world_size']/1.

        #  A A A A A A A A   D D D D D D   V V v V V   E E E E E   R R R R  S     
        # Training the adversarial network stage.
        for epoch in range(start_epoch, epochs):            
            # Apply learning rate warmup
            if epoch < trCf['adv_warmup/epochs']:
                self.G_opt.param_groups[0]['lr']=trCf['G_LR']['init']*float(epoch+1.)/trCf['adv_warmup/epochs']
                self.D_opt.param_groups[0]['lr']=trCf['D_LR']['init']*float(epoch+1.)/trCf['adv_warmup/epochs']

            if epoch==trCf['G_LR']['decay/epochs']:
                self.G_opt.param_groups[0]['lr']=trCf['G_LR']['init']*trCf['G_LR']['gamma']
            if epoch==trCf['D_LR']['decay/epochs']:
                self.D_opt.param_groups[0]['lr']=trCf['D_LR']['init']*trCf['D_LR']['gamma']
                
                
            # Train each epoch for adversarial network.
            Ta=time.time()
            cntD=self.train_adversarial(epoch)
            Tb=time.time()
            # Verify each epoch for adversarial network.
            psnr_value = self.validate( epoch, "adv")
            Tc=time.time()

            ''' OFF
            # Adjust the learning rate of the adversarial model.
            self.D_sched.step()
            self.G_sched.step()
            '''

            # ..... zoo of monitoring and checkpointing is below .....
            #  use running average over last 3 epochs to tag best epoch
            is_best=False
            if  epoch>0.3* epochs:
                psnrC=psnrB
                psnrB=psnrA
                psnrA=psnr_value
                psnrAvr=(psnrA+psnrB+psnrC)/3.
                if best_psnr_value < psnrAvr:
                    is_best =True
                    best_psnr_value = psnrAvr
                    best_epoch=epoch
                    best_cnt=torchD_to_floatD(cntD)
                    best_cnt['val_psnr']=float(psnrAvr)
                    best_cnt['epoch']=int(epoch)
            # Save the weight of the adversarial network under epoch. 
            if self.isRank0 and  epoch% self.params['checkpoint_interval/epochs']==0:
                torch.save(self.D_model.state_dict(), os.path.join(exp_dir2, f"d-epoch{epoch }.pth"))
                torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, f"g-epoch{epoch }.pth"))
            if is_best and self.isRank0:
                torch.save(self.D_model.state_dict(), os.path.join(exp_dir2, "d-best%d.pth"%(epoch)))
                torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "g-best%d.pth"%(epoch)))
                txt='Epoch %d ADV best PSNR=%.2f elaT=%.1f min, exp=%s'%(epoch,best_psnr_value,(time.time()-T0)/60.,self.params['exp_name'])
                self.TBSwriter.add_text('1adv_progress',txt , epoch)
                self.TBSwriter.add_text('2adv_best_cnt',str(best_cnt) , epoch)
                logging.info(txt)
                
            
            if self.isRank0: # monitor LR & speed
                recLR={'G':self.G_opt.param_groups[0]['lr'],'D':self.D_opt.param_groups[0]['lr']}
                #print('JJ:',epoch, recLR)
                recLab='Train_adv/LR '+self.params['exp_name']
                self.TBSwriter.add_scalars(recLab,recLR , epoch)
                Ttot=Tc-Ta
                Ttrain=Tb-Ta
                Tval=Tc-Tb
                rec2={'train':Ttrain,'tot':Ttot,'val':Tval}  # time per epoch
                rec3={'train':kfac*locTrainSamp/Ttrain}  # train glob samp/msec
                if epoch>start_epoch: TperEpoch.append(Ttot)

                rec3.update({'val:20':kfac*locValSamp/Tval/20.})  # val glob samp/msec
                self.TBSwriter.add_scalars("Train_adv/speed_global (samp:sec)",rec3 , epoch)
                self.TBSwriter.add_scalar("Train_adv/epoch_time (sec)", Ttot,epoch)
                
                tV=np.array(TperEpoch)
                if len(tV)>1:
                    tAvr=np.mean(tV); tStd=np.std(tV)/np.sqrt(tV.shape[0])
                else:
                    tAvr=tStd=-1

                txt='ADV epoch %d took %.1f sec, avr=%.2f +/-%.2f sec/epoch, elaT=%.1f min, nGpu=%d, psnr=%.2f best_psnr=%.2f'%(epoch, Ttot, tAvr,tStd,(Tc-T0)/60.,self.params['world_size'],psnr_value,best_psnr_value)
                logging.info(txt)

            
            if self.isRank0: # monitor LR & speed
                txt='ADV epoch %d early-stop=%d,  avr discr(G)=%.2g +-%.2g '%(epoch,cntD['early_stop_discr'],self.earlyStopRing.avr,self.earlyStopRing.std)
                logging.info(txt)
                
            if cntD['early_stop_discr']>0.5:
                if self.isRank0:
                    self.sumRec['early_stop_discr']=True
                    self.sumRec['early_stop_info']=txt
                    self.add_tbsummary_record(txt)
                break  # stop the training
            
        # = = = =  end of Loop-over-advEpochs = = = = = = = 
        if self.isRank0:
            # Save the weight of the adversarial network under the last epoch
            torch.save(self.D_model.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
            torch.save(self.G_model.state_dict(), os.path.join(exp_dir2, "g-last.pth"))
            print('T:done exp_dir2:',exp_dir2)


        if self.isRank0:  # add info to summary, cleanup          
            rec={'epoch_stop':epoch, 'state':'model_trained'} 
            rec['train_duration/sec']=time.time()-T0
            rec['timePerAdverEpoch_sec']=[float('%.2f'%x) for x in [tAvr,tStd] ]
            rec['adv_best_counter']=best_cnt
            self.sumRec.update(rec)

        if self.verb:
            txt='End: exp=%s, best PSNR=%.2f in epoch %d, last epoch %d, numGpu=%d, elapsedTime=%.1f min, globSamp/sec=%.1f globBS=%d' %(self.params['exp_name'],best_psnr_value,best_epoch, epoch,self.params['world_size'],(time.time()-T0)/60.,rec3['train'],self.params['global_batch_size'])
           
            self.add_tbsummary_record(txt)
            logging.info(txt)

            return

        
#...!...!..................
    def pretrain_generator(self, epoch):  # one epoch
        # Pre-training the generator network only.
        tbeg = time.time()
        # Calculate how many iterations there are under epoch.
        batches = len(self.train_loader)
        # Set generator network in training mode.
        self.G_model.train()
        cnt={'pixel_loss':0.}
        for index, (lr, hr) in enumerate(self.train_loader):
            # Copy the data to the specified device.
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            # Initialize the gradient of the generator model.
            if self.params['opt_pytorch']['zerograd']:
                # EXTRA: Use set_to_none option to avoid slow memsets to zero
                self.G_model.zero_grad(set_to_none=True)
            else:
                self.G_model.zero_grad()
                
            # Generate super-resolution images.
            sr = self.G_model(lr)
            
            # Calculate the difference between the super-resolution image and the high-resolution image at the pixel level.
            pixel_loss = self.pixel_criterion(sr, hr)
            # Update the weights of the generator model.
            pixel_loss.backward()
            self.PG_opt.step()
            
            # ..... only monitoring is below
            cnt['pixel_loss']+=pixel_loss
            if self.isRank0:
                if index % self.params['text_log_interval_steps']==0 and self.verb:
                    print(f"Train Epoch[{epoch:04d}]({index:05d}/{batches:05d}) Loss: {pixel_loss:.6f}.")

        # end of epoch, compute summary
        for x in cnt: cnt[x]/=batches

        cnt=all_reduce_dict(cnt,self.dist)

        tend = time.time()
        if self.isRank0:
            self.TBSwriter.add_scalar("Train_gen/Loss_pixel", cnt['pixel_loss'],epoch)
            self.TBSwriter.add_scalar("Train_gen/epoch_time (sec)", (tend-tbeg),epoch)
            
#...!...!..................
    def train_adversarial(self, epoch):  # one epoch
        # Training the adversarial network.

        trCf=self.params['train_conf']
        # Calculate how many iterations there are under Epoch.
        batches = len(self.train_loader)
        cnt={x:0. for x in ['pixel_loss','advers_loss','content_loss','fft_loss','msum_loss','g_loss','d_real','d_fake','d_loss']}
        
        if epoch < trCf['perc_warmup/epochs']:
            percAtten=float(epoch+1.)/trCf['perc_warmup/epochs']
        else:
            percAtten=1.
        
        # Set adversarial network in training mode.
        self.D_model.train()
        self.G_model.train()
        
        # clear example of .detach() logic: https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
        
        for index, (lr, hr) in enumerate(self.train_loader):
            # Copy the data to the specified device.
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            label_size = lr.size(0)
            # Create label. Set the real sample label to 1, and the false sample label to 0.
            real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=self.device)
            fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=self.device)

            # Initialize the gradient of the discriminator model.
            #.... 1. Train D on real+fake
            if self.params['opt_pytorch']['zerograd']:
                self.D_model.zero_grad(set_to_none=True)
            else:
                self.D_model.zero_grad()

            #....  1A: Train D on real
            # Calculate the loss of the discriminator model on the high-resolution image.
            output = self.D_model(hr) # d_real_decision
            d_loss_hr = self.adversarial_criterion(output, real_label) # d_real_error
            d_loss_hr.backward()  # compute/store gradients, but don't change params
            d_hr = output.mean()   # d_real_decision_mean
            
            #....  1B: Train D on fake
            # Generate super-resolution images.
            sr = self.G_model(lr)  # d_fake_data
            # Calculate the loss of the discriminator model on the super-resolution image.
            output = self.D_model(sr.detach()) # d_fake_decision, detach to avoid training G on these labels 
            d_loss_sr = self.adversarial_criterion(output, fake_label) # d_fake_error 
            d_loss_sr.backward()
            d_sr1 = output.mean()  # d_fake_decision_mean
            # Update the weights of the discriminator model.
            d_loss = d_loss_hr + d_loss_sr
            self.D_opt.step() # Only optimizes D's parameters; changes based on stored gradients from backward()


            # Initialize the gradient of the generator model.
            if self.params['opt_pytorch']['zerograd']:
                self.G_model.zero_grad(set_to_none=True)
            else: 
                self.G_model.zero_grad()

            #.... 2. Train G on D's response (but DO NOT train D on these labels)
            # Calculate the loss of the discriminator model on the super-resolution image.
            output = self.D_model(sr) # dg_fake_decision, len=BS filled w/ probabilities of being real

            # Perceptual_loss= weighted sum:  pixel + content +  adversarial + power_spect
            advers_loss =  trCf['advers_weight'] *self.adversarial_criterion(output, real_label) # will train G to pretend it's genuine
            content_loss =  trCf['content_weight'] *self.content_criterion(sr, hr)
            pixel_loss  =  percAtten *trCf['pixel_weight'] *self.pixel_criterion(sr, hr)#.detach())

            # convert from log(mass+1 ) --> mass+1, dim=[BS,1,512,512]
            hr_field=transf_img2field_torch(hr)
            sr_field=transf_img2field_torch(sr)

            #  compute integral for fields difference (more accurate?)
            delta_msum=torch.sum(hr_field-sr_field,(2,3))
            #print('bb',hr_msum.shape,type(hr_msum))
            
            msum_loss= percAtten *trCf['msum_weight'] *self.msum_criterion(delta_msum, torch.zeros_like(delta_msum))
            
            hr_fft=compute_fft( hr_field)
            sr_fft=compute_fft( sr_field)
            fft_loss = percAtten *trCf['fft_weight'] *self.fft_criterion(sr_fft, hr_fft)
                            
            # Update the weights of the generator model.
            g_loss =  advers_loss +  pixel_loss + content_loss + fft_loss + msum_loss
            
            g_loss.backward() 
            self.G_opt.step()  # Only optimizes G's parameters
            
            d_sr2 = output.mean()  #dg_fake_decision_mean, what is the purpose of 'sr2' ???
            
            # ..... only monitoring is below + early-stop condition(s)
            cnt['d_real']+=d_hr
            cnt['d_fake']+=d_sr1
            cnt['d_loss']+=d_loss
            cnt['pixel_loss']+=pixel_loss
            cnt['fft_loss']+=fft_loss
            cnt['msum_loss']+=msum_loss
            cnt['advers_loss']+=advers_loss
            cnt['content_loss']+=content_loss
            cnt['g_loss']+=g_loss

            if self.isRank0:
                if index % self.params['text_log_interval_steps']==0 and self.verb:
                    print(f"Train stage: adversarial "
                          f"Epoch[{epoch + 1:04d}]({index + 1:05d}/{batches:05d}) "
                          f"D Loss: {d_loss.item():.3g} G Loss: {g_loss.item():.3g} "
                          f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.3g}/{d_sr2:.3g}.")
        # end of epoch, compute summary
        for x in cnt: cnt[x]/=batches        
        
        #print('bef',self.params['world_rank'],cnt)
        cnt=all_reduce_dict(cnt,self.dist)
        self.earlyStopRing.update(cnt['d_fake'])
        #print('RR',self.params['world_rank'],cnt['d_fake'],self.earlyStopRing.buf)
        cnt['early_stop_discr']=torch.tensor(self.earlyStopRing.check(),dtype=torch.float32)
        if self.isRank0:

            rec={'percept_w_atten':percAtten,'avr+std_dec_G':self.earlyStopRing.avr+self.earlyStopRing.std}
            self.TBSwriter.add_scalars("Train_adv/misc", rec,epoch)
           
            for x in ['g_loss','d_loss']:
                self.TBSwriter.add_scalar("Train_adv/"+x, cnt[x],epoch)
                
            rec={x:cnt[x] for x in ['pixel_loss','advers_loss','content_loss','fft_loss','msum_loss'] }
            rec['sum']=cnt ['g_loss']
            self.TBSwriter.add_scalars("Train_adv/g_crit", rec, epoch)

            rec={x:cnt[x] for x in ['d_real','d_fake'] }
            self.TBSwriter.add_scalars("Train_adv/d_decision", rec, epoch)
        return cnt
            
#...!...!..................
    def validate(self, epoch, stage) -> float:
        """ Verify the generator model on validation data
            stage is [gen,adv] toredirect the printouts.
        Returns: PSNR value(float).
        """
        # Calculate how many iterations there are under epoch.
        batches = len(self.valid_loader)
        # Set generator model in verification mode.
        self.G_model.eval()
        with torch.no_grad():
            cnt={'psnr':0.}
            for index, (lr, hr) in enumerate(self.valid_loader):
                # Copy the data to the specified device.
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                # Generate super-resolution images.
                sr = self.G_model(lr)
                # Calculate the PSNR indicator.
                mse_loss = self.psnr_criterion(sr, hr)
                psnr_value = 10 * torch.log10(1 / mse_loss)                
                cnt['psnr']+=psnr_value
               
            # end of epoch, compute summary
            for x in cnt: cnt[x]/=batches
            
            cnt=all_reduce_dict(cnt,self.dist)

            # Write the value of each round of verification indicators into Tensorboard.
            if self.isRank0:
                self.TBSwriter.add_scalar("Train_%s/Val_PSNR"%stage,cnt['psnr'] , epoch)
                
                if self.params['pred_dump_interval/epochs']:
                    if epoch% self.params['pred_dump_interval/epochs']==0:
                        outF=os.path.join(self.params['val_mon_path'],'valid-%s-epoch%d.h5'%(stage,epoch))
                        logging.info('pred dump '+outF)
                        bigD={'lr':lr,'sr':sr,'hr':hr}                        
                        # convert images to fileds
                        for x in bigD:
                            y=bigD[x]
                            z=y.detach().cpu().numpy()
                            bigD[x]=np.exp(z).astype(np.float32)
                        metaD={}
                        for x in ['sim3d','field2d']:
                            metaD[x]=self.params[x]
                        metaD['domain']='valid_batch'
                        metaD['exp_name']=self.params['exp_name']
                        metaD['numSamples']=int(lr.shape[0])
                        metaD['modelDesign']=self.params['myId']
                        metaD['model_path']='train dump'
                        metaD['gen_sol']='%s-epoch%d'%(stage,epoch)
                        
                        #print('DDD');pprint(metaD)
                        write3_data_hdf5(bigD,outF,metaD=metaD,verb=0)

        return cnt['psnr']
