from toolbox.tmp_figures import ( plt_slices,  plt_power )
import logging  # text-to-screen
from torch.utils.tensorboard import SummaryWriter
import os

#  Util class collecting all TB-SummaryWriter related functionality

#............................
#............................
#............................
class TBSwriter(object):
    def __init__(self,expDir):
        #print('TBS:cstr, expDir=',expDir)
        self.TBSwriter=SummaryWriter(os.path.join(expDir, 'tb'))
        self.tbsummary_step=0

#...!...!..................
    def add_tbsummary_record(self,txt):
        self.TBSwriter.add_text('0summary',txt , global_step=self.tbsummary_step)
        self.tbsummary_step+=1

# - - - - -  NOT  USED  YET - - - - - -         
#...!...!..................
    def TBlog_epoch_train(self,epoch_loss,inpCube,outCube,tgtCube,timeD,speedD):
        
        epoch=self.epoch
        self.TBSwriter.add_scalars('epoch time (sec) ',timeD , epoch)
        self.TBSwriter.add_scalars('glob_speed (samp:sec) ',speedD , epoch)

        self.TBSwriter.add_scalar('loss_G',epoch_loss[0] ,epoch)
        
        recLR={'G':self.G_opt.param_groups[0]['lr'],'D':self.D_opt.param_groups[0]['lr']}
        self.TBSwriter.add_scalars('LR',recLR , epoch)

        if self.doDpass:
            self.TBSwriter.add_scalar('loss_D',epoch_loss[1] ,epoch)
            self.TBSwriter.add_scalar('noise_D', self.D_noise_std,epoch)
            self.TBSwriter.add_scalars('lossTFR',
                { 'total': epoch_loss[2],
                    'fake': epoch_loss[3],
                    'real': epoch_loss[4],},
                global_step=epoch,
            )
            
        skip_chan = 0
        #if args.adv and epoch >= args.adv_start and args.cgan:
        #    skip_chan = sum(args.in_chan)
    
        fig_slices = plt_slices(
            inpCube[-1], outCube[-1, skip_chan:], tgtCube[-1, skip_chan:],
            outCube[-1, skip_chan:] - tgtCube[-1, skip_chan:],
            title=['inp', 'out', 'tgt', 'residues'],
            #**args.misc_kwargs,
        )
        self.TBSwriter.add_figure('fig_slices', fig_slices, global_step=epoch)
        fig_slices.clf()

        fig1,fig2,fig3 = plt_power(
            inpCube, outCube[:, skip_chan:], tgtCube[:, skip_chan:],
            label=['inp', 'out', 'tgt'] )
        self.TBSwriter.add_figure('fig_displ/P', fig1, epoch)
        self.TBSwriter.add_figure('fig_displ/relP', fig2,epoch)
        self.TBSwriter.add_figure('fig_displ/P*k^3', fig3, epoch)
        fig1.clf();  fig2.clf(); fig3.clf();

        fig4,fig5,fig6 = plt_power(1.0,
                    dis=[inpCube, outCube[:, skip_chan:], tgtCube[:, skip_chan:]],
                    label=['inp', 'out', 'tgt'])
        self.TBSwriter.add_figure('fig_dens/P', fig4, epoch)
        self.TBSwriter.add_figure('fig_dens/relP', fig5,epoch)
        self.TBSwriter.add_figure('fig_dens/P*k^3', fig6, epoch)
        fig4.clf();  fig5.clf(); fig6.clf();

        logging.info('Epoch: %2d, plots created '%(epoch))

#...!...!..................  OFF: line 354
    def TBlog_G_step_train(self,loss,grads):
        
        gstep=self.iterD['gcnt']
        self.TBSwriter.add_scalar('5-G_loss/batch train MSE(out-tgt) (kpc/h)^2', loss.item(), global_step=gstep)
        
        self.TBSwriter.add_scalar('5-G_grad/first_layer train', grads[0], global_step=gstep)
        self.TBSwriter.add_scalar('5-G_grad/last_layer train', grads[-1], global_step=gstep)


#...!...!.................. OFF: line 450
    def TBlog_D_train_step(self,loss_adv,adv_loss_tot,adv_loss_fake,adv_loss_real, adv_grads):
        gstep=self.iterD['gcnt']
        self.TBSwriter.add_scalar('4-loss_per_batch train/adv/G', loss_adv.item(),global_step=gstep)
        self.TBSwriter.add_scalars('4-loss_per_batch train/adv/D',
            { 'total': adv_loss_tot.item(),
                'fake': adv_loss_fake.item(),
                'real': adv_loss_real.item(),},
            global_step=gstep,
        )

        self.TBSwriter.add_scalar('3-grad/adv/first', adv_grads[0], global_step=gstep)
        self.TBSwriter.add_scalar('3-grad/adv/last', adv_grads[-1], global_step=gstep)
      

