import torch
import torch.fft # needed by pytorch 1.7


#...!...!..................
def custom_LR_scheduleB(epoch,opt,lrCf,maxEpoch):
    # linear warmup and linear decay
    if epoch<lrCf['warmup/epochs']:        
        opt.param_groups[0]['lr']=lrCf['init']*float(epoch+1.)/lrCf['warmup/epochs']
        return
    if epoch>lrCf['decay/epochs']:
        totEpoch=maxEpoch - lrCf['decay/epochs']
        xEpoch=epoch - lrCf['decay/epochs']
        opt.param_groups[0]['lr']=lrCf['init']*(1.-(1-lrCf['reduce'])*xEpoch/totEpoch )
        return
        
        
#...!...!..................
def integral_loss_func(field1,field2):
    delta=field1-field2
    msum=torch.sum(delta,(2,3))
    msum_loss= torch.log(torch.linalg.norm(msum))
    return msum_loss


#...!...!..................
def torchD_to_floatD(inp):
    out={}
    for x in inp:
        y=inp[x].detach().cpu().numpy()
        out[x]=float(y)
    return out
    
#...!...!..................
def transf_field2img_torch(field):
    return torch.log(field)

#...!...!..................
def transf_img2field_torch(img):
    return torch.exp(img)

#...!...!..................
def compute_fft(fieldBatch):  # used in training loss
    #assert fieldB.shape[1]==1 # 1 channel
    fourier_image = torch.fft.fftn(fieldBatch) #FFTs only the last two dimensions by default.
    #print('FTCS:inp',fieldBatch.shape,'fft:',fourier_image.shape)
    # >>> torch.Size([16, 1, 512, 512]) fft: torch.Size([16, 1, 512, 512])
    dimEuc=fieldBatch.shape[-1]
    # clip FFT image to a quadrant
    dimFft=dimEuc//2
    fourier_image=fourier_image[:,:,:dimFft,:dimFft]    
    fourier_amplitudes2= torch.abs(fourier_image)**2
    #print('FFT fourier_amplitudes2',fourier_amplitudes2.shape)
    return torch.log(fourier_amplitudes2+1.)

#...!...!..................
def all_reduce_dict(input_dict, dist,average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    #for x in input_dict: print('AR:bef=',x,type(input_dict[x]),input_dict[x])
    if dist==None: return input_dict
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = []
        # sort the keys so that they are consistent across processes
        for k in names:
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict 
