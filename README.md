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

Training:
	speedup 25% 8h -> 6h, can be further speedup by using caching
Testing(without fuse bn):
	     mAP 	        FPS
        21.8 -> 21.9         85 -> 89      for batchsize 8
	21.2 -> 21.3  	   100 -> 102 	   for batchsize 1

We can get several things from the result:
1. data loader is the bottleneck when batchsize is large for the resnet18 backbone on my 2 core 4 threading cpu...
   By using powerful cpu or DALI as dataloader, my batch-version inference will be further boost.
2. One interesting thing is that the mAP of batchsize 8 which is same as training batchsize is higher the batchsize 1.
   This implies that the align method for stacking the images brings **`inductive bias`**.
   By changing the default top-left alignment to center alignment or random alignment by designing a special **`collate`** function
   for dataloader when stacking the images will get about 0.5 mAP improvement for realtime batchsize 1 inference.
3. Last but not the least, all of this conclusions apply for CenterNet* and CenterNet2 which have 7 mAP improvement in the same setting.
   The soft guassian heatmap and the OHEM sampling strategy attributes a lot for its faster convergence and higher mAP on small backbone.
   Also, I have not tested the BiFPN, hmm...
   By integrating the once-for-all modeling and training strategy, hmm...
   Replacing GN with sycnBN which is more inference friendly is also a good chioce, do you think so?
   
------------------------

TODO: report FPS with single 1080Ti and asyncio with 3 cuda stream for dataloader and model
      when server is avaliable...
