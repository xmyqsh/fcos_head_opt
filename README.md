# FCOS HEAD OPT

This opt version of FCOS can run 25% faster on resnet18 backbone with
batch size 16 on 2 1080Ti.

It can further speed up when use more powerful/more threading cpu for dataloading and larger batchsize for training.

The large batchsize inference also speeduped if the dataloading is not
the bottleneck.

Same speedup logit of training and testing is orthognal for CenterNet\*
and CenterNet2.
So the training of CenterNet\* and CenterNet2 can also be speeduped more than 25%.

This is very meaningful for the next work: `once-for-all-CenterNet2`.

----------------------

This is not the latest code which may have small bug.
The code will be updated as long as the remote server is avaliable.
