# I'm doing a solar forecasting using satellite image, what's the best approach and model design to learn the cloud movement?

For learning cloud movement from satellite imagery, the strongest practical approach is usually a spatiotemporal nowcasting model that predicts future cloud fields first, then maps those forecasts to irradiance or power. If your horizon is short, roughly 0 to 2 hours, cloud advection dominates, so a motion-aware video prediction model works better than a pure tabular regressor.

A good design is a hybrid encoder-decoder:

1. Input a sequence of past satellite frames, typically 6 to 24 frames at 5 to 15 minute cadence.
2. Use a spatial encoder, usually a CNN or U-Net style encoder, on each frame.
3. Use a temporal motion module to learn how clouds move:
   - For a strong baseline: ConvLSTM or TrajGRU
   - For better performance: a temporal transformer with spatial tokens
   - For the most practical setup: learned flow or warping plus ConvLSTM/Transformer
4. Decode future cloud states or future satellite images with a U-Net decoder.
5. Add a second head to predict site-level targets such as GHI, DNI, or PV power.

If you want one model recommendation: start with a U-Net encoder-decoder plus ConvLSTM bottleneck and an explicit warping head. That is usually the best balance of accuracy, stability, and training cost.

Use targets that make the physics easier. Instead of predicting raw power directly, predict clear-sky index $k^* = \frac{GHI}{GHI_{clear}}$ or a cloud/irradiance map, then convert that to power using a separate PV model. This reduces the burden on the network because solar geometry is handled outside the motion model.

The most useful inputs are:
- Multi-channel geostationary satellite imagery, not just visible
- Solar zenith and azimuth angles
- Time of day and day of year
- Site location and elevation
- Optional weather/NWP features for horizons beyond 2 hours

For losses, do not rely on plain MSE alone. Use a multi-task loss:
- Image reconstruction loss on future cloud fields
- Motion consistency or warping loss between predicted and advected clouds
- Regression loss on irradiance or power
- Quantile loss if you want uncertainty bands

A strong training strategy is to predict both:
- Future cloud mask or cloud optical proxy
- Future irradiance or PV output

That auxiliary cloud task usually helps the model learn real movement instead of just memorizing output patterns.

Model choice by forecast horizon:
- 0 to 60 min: optical flow baseline, ConvLSTM, TrajGRU, or warping U-Net
- 1 to 3 hours: temporal transformer or hybrid ConvLSTM-transformer
- 3 to 6+ hours: combine satellite model with NWP; pure image models usually degrade

A practical stack to build is:
- Baseline 1: persistence
- Baseline 2: optical flow / dense advection
- Main model: U-Net + ConvLSTM + warping
- Advanced model: spatiotemporal transformer with learned motion field

If you are optimizing for research quality, the key idea is this: cloud movement is mostly an advection problem, so your architecture should explicitly model motion, not just sequence-to-value regression. Pure CNN-to-MLP models usually underperform because they do not preserve the spatial dynamics.

# sketch a concrete architecture with 18 step of 10 min data to forecast 18 steps of 10 min data

Use an encoder-forecast-decoder nowcasting model that takes 18 past frames and outputs 18 future frames, with an auxiliary site-level irradiance head. For this problem, I would start with a motion-aware U-Net plus ConvLSTM bottleneck. It is simpler and more stable than a full video transformer, but still learns cloud advection well.

**Recommended Architecture**

Input:
- Sequence length: 18 past steps
- Horizon: 18 future steps
- Cadence: 10 min
- Tensor: $X \in \mathbb{R}^{B \times 18 \times C \times H \times W}$
- Typical channels:
  - visible, IR, water vapor, cloud mask if available
  - solar zenith angle
  - solar azimuth encoded as $\sin$ and $\cos$
  - optional clear-sky irradiance map
- If solar angles are scalars, broadcast them to $H \times W$

Backbone:
1. Frame encoder
   - Shared 2D CNN/U-Net encoder applied to each of the 18 frames
   - Example channels:
     - stage 1: 64
     - stage 2: 128
     - stage 3: 256
     - bottleneck: 512
   - Output per frame:
     - low-level skip features for decoder
     - compressed latent map $z_t \in \mathbb{R}^{B \times 512 \times H/16 \times W/16}$

2. Temporal motion module
   - Stack the 18 latent maps over time
   - Pass through 2 to 3 ConvLSTM layers:
     - ConvLSTM(512 -> 512, kernel 3)
     - ConvLSTM(512 -> 512, kernel 3)
   - Final hidden state summarizes motion history
   - This is the core cloud-movement learner

3. Autoregressive forecast module
   - Predict 18 future latent states one at a time
   - At each future step:
     - use previous hidden state
     - optionally use previous predicted latent
     - output next latent $\hat{z}_{t+1}$
   - Better variant:
     - predict a flow field $F_{t+1} \in \mathbb{R}^{B \times 2 \times H/16 \times W/16}$
     - warp previous latent using differentiable sampling
     - refine with residual block
   - This explicit warp helps cloud advection a lot

4. Decoder
   - U-Net style decoder upsamples each future latent to image space
   - Use encoder skip features from the last observed frame, or learned static context
   - Output one of:
     - future cloud image channels
     - future cloud probability map
     - future clear-sky index map

5. Site forecast head
   - For a plant/site location, crop a small patch around the site from decoded future maps
   - Apply spatial pooling plus MLP
   - Output:
     - $\hat{y} \in \mathbb{R}^{B \times 18}$ for GHI, CSI, or PV power
   - If multiple sites, use a coordinate-conditioned head

**Concrete Tensor Flow**

Given $B \times 18 \times C \times 256 \times 256$:

1. Per-frame encoder:
   - each frame -> latent $512 \times 16 \times 16$
2. Stack time:
   - $B \times 18 \times 512 \times 16 \times 16$
3. ConvLSTM:
   - output future latent sequence
   - $B \times 18 \times 512 \times 16 \times 16$
4. Decoder:
   - each future latent -> predicted map
   - for example $B \times 18 \times 1 \times 256 \times 256$
5. Site head:
   - from each future map -> scalar forecast
   - $B \times 18$

**Best Target Choice**

Do not predict PV power directly as the only target. Predict:
- future cloud-related image target, and
- future site-level clear-sky index or GHI

Best practical setup:
- image head predicts future clear-sky index map
- site head predicts future site clear-sky index
- then convert to power with a separate PV conversion model

Reason:
- cloud motion is spatial
- solar geometry and plant conversion are easier to handle outside the motion model

**Loss Function**

Use a multi-task loss:
$$
\mathcal{L} =
\lambda_1 \mathcal{L}_{img}
+ \lambda_2 \mathcal{L}_{site}
+ \lambda_3 \mathcal{L}_{grad}
+ \lambda_4 \mathcal{L}_{warp}
$$

Where:
- $\mathcal{L}_{img}$: MAE or Huber on future maps
- $\mathcal{L}_{site}$: MAE on 18-step site forecast
- $\mathcal{L}_{grad}$: edge/gradient loss to preserve cloud boundaries
- $\mathcal{L}_{warp}$: consistency loss between warped and predicted latent/image

A good starting point:
- $\lambda_1 = 1.0$
- $\lambda_2 = 2.0$
- $\lambda_3 = 0.2$
- $\lambda_4 = 0.2$

Use MAE or Huber, not only MSE. MSE tends to blur cloud structures.

**Training Strategy**

Sequence setup:
- input: $t-17$ to $t$
- target: $t+1$ to $t+18$

Teacher forcing:
- During early training, feed true previous future latent occasionally
- Then reduce teacher forcing over time

Normalization:
- Normalize each channel separately
- For visible channels, account for solar-angle dependence
- For site target, prefer clear-sky index:
$$
k_t^* = \frac{GHI_t}{GHI_{clear,t}}
$$

Sampling:
- Oversample rapidly changing cloud periods
- Include sunrise/sunset carefully, or train separate daylight-only model

**Recommended Hyperparameters**

Starter configuration:
- Input resolution: 128x128 or 256x256 around the site
- Batch size: 4 to 16 depending on GPU
- Optimizer: AdamW
- Learning rate: $1 \times 10^{-4}$
- Weight decay: $1 \times 10^{-4}$
- Epochs: 50 to 150
- Scheduler: cosine decay or ReduceLROnPlateau

Model sizes:
- Encoder channels: 64, 128, 256, 512
- ConvLSTM layers: 2
- ConvLSTM hidden dim: 512
- Decoder mirrors encoder

**Strong Baselines To Compare Against**

You should always compare against:
1. Persistence
   - future = last observed condition
2. Optical flow advection
   - estimate motion from recent satellite frames and advect clouds
3. ConvLSTM without explicit warping
4. Your full model with warping

If the full model does not beat optical flow on short horizons, the architecture is not learning motion well enough.

**Practical Design Recommendation**

If you want one concrete model to implement first, use this:

- Shared CNN encoder on 18 past frames
- 2-layer ConvLSTM bottleneck
- Autoregressive 18-step latent forecasting
- Learned flow-based warping at latent resolution
- U-Net decoder to future clear-sky index maps
- MLP head for 18-step site forecast

That is the best balance of:
- motion awareness
- implementation cost
- training stability
- forecast skill for 10-minute nowcasting

**Minimal PyTorch Module Breakdown**

You would implement:
1. `FrameEncoder`
2. `ConvLSTMForecaster`
3. `LatentWarpBlock`
4. `FrameDecoder`
5. `SiteHead`
6. `NowcastModel`

Forward pass:
- encode 18 frames
- run temporal encoder
- autoregress 18 future latents
- decode future maps
- pool around site
- predict 18-step sequence

If you want, I can next give you either:
1. a layer-by-layer PyTorch architecture with tensor shapes,
2. a training loop and loss implementation, or
3. a simpler transformer alternative for the same 18-to-18 setup.


# what is warp? how to calculate warp loss?

In this setting, warp means spatially moving an image or latent feature map according to a learned motion field. The model predicts where each pixel or feature should move between time steps, then uses that motion field to shift the cloud pattern instead of redrawing the whole next frame from scratch.

For clouds, this matches the physics reasonably well: most short-horizon evolution is advection, meaning the cloud field mostly moves, with some growth and decay. Warping gives the model an explicit way to represent that motion.

If you have a frame $I_t$ and want to predict $I_{t+1}$, the model can predict a flow field

$$
F_t(x, y) = (u(x,y), v(x,y))
$$

where $u$ and $v$ are horizontal and vertical displacements. The warped frame is sampled from the previous frame:

$$
\hat{I}_{t+1}^{warp}(x,y) = I_t(x - u(x,y),\ y - v(x,y))
$$

In practice, you do not index pixels directly. You use differentiable sampling, typically `grid_sample` in PyTorch, so gradients flow through the motion field.

There are two common places to warp:

1. Image-space warp  
Warp the actual satellite image or cloud mask.

2. Latent-space warp  
Warp encoder features, which is often easier and more stable because the latent maps are smaller and smoother.

For model design, the usual pattern is:

1. Encode past frames into latent features.
2. Predict a flow field from the current hidden state.
3. Warp the last latent using that flow.
4. Optionally add a residual refinement block.
5. Decode to the next future frame.

A simple formulation is

$$
\hat{z}_{t+1} = \text{Refine}(\text{Warp}(z_t, F_t))
$$

where $z_t$ is the latent feature map.

**Warp loss**

Warp loss measures whether the predicted motion field moves the previous frame or latent into something close to the next true frame or latent.

The simplest image-space warp loss is:

$$
\mathcal{L}_{warp} = \left\| I_{t+1} - \text{Warp}(I_t, F_t) \right\|_1
$$

You can also use Huber loss instead of $L_1$.

If you warp latent features instead of images:

$$
\mathcal{L}_{warp}^{latent} = \left\| z_{t+1} - \text{Warp}(z_t, F_t) \right\|_1
$$

That requires either true future latents from the encoder or a teacher-forced latent target during training.

For cloud nowcasting, a stronger version is to combine several terms:

$$
\mathcal{L}_{warp-total} =
\lambda_1 \mathcal{L}_{photo} +
\lambda_2 \mathcal{L}_{smooth} +
\lambda_3 \mathcal{L}_{grad}
$$

with:

Photometric or reconstruction term:
$$
\mathcal{L}_{photo} = \left\| I_{t+1} - \text{Warp}(I_t, F_t) \right\|_1
$$

Flow smoothness term:
$$
\mathcal{L}_{smooth} =
\left\| \nabla_x u \right\|_1 +
\left\| \nabla_y u \right\|_1 +
\left\| \nabla_x v \right\|_1 +
\left\| \nabla_y v \right\|_1
$$

This discourages noisy, unrealistic motion fields.

Gradient or edge consistency term:
$$
\mathcal{L}_{grad} =
\left\| \nabla I_{t+1} - \nabla \text{Warp}(I_t, F_t) \right\|_1
$$

This helps preserve cloud boundaries instead of learning blurry motion.

A very practical starting loss is:

$$
\mathcal{L}_{warp-total} =
\left\| I_{t+1} - \text{Warp}(I_t, F_t) \right\|_1
+ 0.1 \cdot \mathcal{L}_{smooth}
+ 0.1 \cdot \mathcal{L}_{grad}
$$

If your model predicts 18 future steps, compute this per step and average:

$$
\mathcal{L}_{warp-seq} =
\frac{1}{18}\sum_{k=1}^{18}
\left\| I_{t+k} - \text{Warp}(\tilde{I}_{t+k-1}, F_{t+k-1}) \right\|_1
$$

Here $\tilde{I}_{t+k-1}$ is either:
1. the true previous frame during teacher forcing, or
2. the model's previous prediction during autoregressive rollout.

**Important caveat**

Warp alone is not enough. Clouds do not only move; they also form, dissipate, and change shape. So most good models use:

$$
\hat{I}_{t+1} = \text{Warp}(I_t, F_t) + R_t
$$

where $R_t$ is a residual correction predicted by the network. Then you supervise the final prediction with a normal forecast loss, and optionally supervise the pure warp with a separate warp loss.

That gives you:
- motion from the warp term
- cloud growth/decay from the residual term

**PyTorch-style intuition**

Conceptually:

```python
flow = flow_head(hidden)              # [B, 2, H, W]
warped = grid_sample(prev_frame, flow_to_grid(flow))
pred = warped + residual_head(hidden)

loss_warp = l1_loss(warped, target_next_frame)
loss_pred = l1_loss(pred, target_next_frame)
loss = loss_pred + 0.2 * loss_warp + 0.1 * smoothness(flow)
```

So:
- `loss_warp` teaches motion
- `loss_pred` teaches final forecasting
- `smoothness(flow)` keeps the flow realistic

**When to use image-space vs latent-space warp**

Use image-space warp if:
- you want direct interpretability
- your channels are physically meaningful
- your resolution is manageable

Use latent-space warp if:
- images are large
- raw pixel alignment is noisy
- you want a more stable training signal

For satellite cloud forecasting, I would usually start with latent-space warp in the model and image-space warp loss as an auxiliary term if decoding is stable.

**Practical recommendation**

For your 18-to-18 model, start with:
1. predict flow at each future step
2. warp the previous latent
3. add residual refinement
4. decode the final future image
5. use total loss:
   - forecast loss on decoded frame
   - warp loss on warped frame
   - smoothness on flow

A solid first formula is:

$$
\mathcal{L} =
\mathcal{L}_{forecast}
+ 0.2 \mathcal{L}_{warp}
+ 0.05 \mathcal{L}_{smooth}
$$

If you want, I can give you the exact PyTorch implementation of:
1. `warp()` using `grid_sample`
2. `smoothness(flow)`
3. a full `warp_loss()` function for your 18-step forecast model.


# I need to forecast 18 steps, why u mentioned to 'decode the final future image'?

“Decode the final future image” was imprecise wording. For an 18-step forecast, you normally decode a future image at every forecast step, not only the last one.

What I meant was:

- At each future step, the model forms an intermediate motion-based state:
  $$
  \tilde{z}_{t+k} = \mathrm{Warp}(z_{t+k-1}, F_{t+k-1})
  $$
- Then it refines that state:
  $$
  z_{t+k} = \mathrm{Refine}(\tilde{z}_{t+k})
  $$
- Then it decodes that step’s latent into that step’s predicted image:
  $$
  \hat{I}_{t+k} = \mathrm{Decoder}(z_{t+k})
  $$

So for 18-step forecasting, the output is:

$$
\hat{I}_{t+1}, \hat{I}_{t+2}, \dots, \hat{I}_{t+18}
$$

not just $\hat{I}_{t+18}$.

The word “final” was referring to the final prediction for one step after warp plus residual refinement, not the final horizon only. A better phrasing would have been:

- warp previous state
- refine it
- decode the current future step
- repeat for all 18 steps

There are two valid design choices.

**Option 1: Decode every future step**
This is the standard choice for your problem.

Pipeline:
1. Encode 18 past images
2. Forecast latent for step 1
3. Decode image for step 1
4. Forecast latent for step 2
5. Decode image for step 2
6. Continue until step 18

This gives direct supervision at every horizon:
$$
\mathcal{L}_{forecast} = \frac{1}{18}\sum_{k=1}^{18}\left\|\hat{I}_{t+k} - I_{t+k}\right\|
$$

This is what you want if your target is 18 future satellite or cloud maps.

**Option 2: Decode only the last step**
This is only appropriate if your task is specifically “given the past 18 steps, predict only the image at +180 min.” That is not your setup.

For your case, if you need all 18 future steps, decode all 18.

A concrete sequence-to-sequence structure for your use case is:

- Input:
  $$
  X = [I_{t-17}, I_{t-16}, \dots, I_t]
  $$
- Encoder produces latent history:
  $$
  [z_{t-17}, \dots, z_t]
  $$
- Temporal module summarizes motion
- Forecast loop for $k = 1,\dots,18$:
  - predict flow $F_{t+k-1}$
  - warp previous latent
  - refine latent
  - decode image $\hat{I}_{t+k}$

So the decoder is used 18 times, once per forecast step.

If your final target is not future images but site irradiance or PV power, there are again two choices:

1. Predict 18 future images/maps and then derive 18 site values from them
2. Predict 18 latent states and attach a regression head at each step

In either case, it is still an 18-step decoder or 18-step output head, not a single final-step decoder.

The cleanest architecture for your task is:

- Past 18 frames in
- Future 18 latents rolled out autoregressively
- Future 18 images decoded, one per step
- Optional future 18 irradiance/power values from each decoded map

If you want, I can draw the exact 18-in to 18-out dataflow as a block diagram and give you the tensor shapes at each stage.


# how to design refine block?

The refine block corrects what pure warping cannot model well: cloud growth, decay, deformation, boundary sharpening, and small flow errors. Its job is not to relearn the whole next frame. Its job is to make a residual correction on top of the warped state.

The clean formulation is:

$$
\tilde{z}_{t+k} = \mathrm{Warp}(z_{t+k-1}, F_{t+k-1})
$$

$$
r_{t+k} = \mathrm{RefineBlock}(\tilde{z}_{t+k}, h_{t+k-1}, c_{t+k})
$$

$$
z_{t+k} = \tilde{z}_{t+k} + r_{t+k}
$$

where:
- $\tilde{z}_{t+k}$ is the warped latent
- $F_{t+k-1}$ is the predicted flow
- $h_{t+k-1}$ is temporal hidden state from ConvLSTM or transformer
- $c_{t+k}$ is optional conditioning such as solar angle or time embedding
- $r_{t+k}$ is a residual correction

That residual form matters. It keeps the block focused and makes training more stable.

**Best practical refine block**

For a first strong model, use a small residual CNN block at latent resolution.

Input:
- warped latent: $\tilde{z} \in \mathbb{R}^{B \times D \times H \times W}$
- optional temporal hidden state: $h \in \mathbb{R}^{B \times D_h \times H \times W}$

Concatenate:
$$
x = [\tilde{z}, h]
$$

Then use:

1. `3x3 conv` -> `GroupNorm` or `BatchNorm` -> `GELU`
2. `3x3 conv` -> `GroupNorm` -> `GELU`
3. `3x3 conv` -> residual output with channel size $D$

Then:
$$
z_{next} = \tilde{z} + \alpha \cdot r
$$

with $\alpha = 1$ initially, or even a learnable scalar initialized small like $0.1$.

This is the simplest good version.

A concrete channel design if latent dimension is 256:
- input channels: 256 warped + 256 hidden = 512
- conv1: 512 -> 256
- conv2: 256 -> 256
- conv3: 256 -> 256

So the refine block is shallow and local. That is usually enough because the motion has already been handled by the warp.

**Recommended block structure**

A good default is:

```text
concat(warped_latent, hidden_state)
-> 3x3 conv
-> group norm
-> GELU
-> 3x3 conv
-> group norm
-> GELU
-> 3x3 conv
-> residual correction
-> add to warped_latent
```

If you want slightly better performance, add a channel attention or squeeze-excitation unit after the second conv. But do not start there unless needed.

**Why residual design is better**

If you predict the whole next latent directly:
$$
z_{t+k} = \mathrm{RefineBlock}(\tilde{z}_{t+k}, h_{t+k-1})
$$

the network may ignore the warp path and just learn blurry frame synthesis. Residual refinement forces the model to use motion first and correction second.

That is the right inductive bias for cloud advection.

**What should the refine block see?**

Best inputs:
1. warped latent $\tilde{z}_{t+k}$
2. temporal hidden state $h_{t+k-1}$
3. optional last observed latent $z_t$ or previous predicted latent
4. optional time features:
   - forecast step index
   - solar zenith
   - solar azimuth sine/cosine

A strong version is:

$$
r_{t+k} = \mathrm{RefineBlock}([\tilde{z}_{t+k}, h_{t+k-1}, e_k])
$$

where $e_k$ is a broadcast future-step embedding.

That helps because cloud appearance depends on illumination as well as motion.

**How deep should it be?**

Do not make it too deep at first.

Good starting point:
- 2 to 3 residual conv layers
- kernel size 3
- same spatial size as latent
- no downsampling

Reason:
- the refine block should fix local errors
- large-scale motion should already come from the flow/warp module

If the refine block is too large, it starts competing with the motion model.

**Should refine block operate in image space or latent space?**

Use latent space.

Reason:
- cheaper
- smoother
- easier to train
- less sensitive to raw image noise

Image-space refinement can be added later as a final decoder-side residual, but latent-space refinement is the right first design.

**A stronger version: gated refine block**

If you want the model to decide how much correction to apply, predict both a residual and a gate:

$$
g = \sigma(\mathrm{Conv}(x))
$$

$$
r = \mathrm{Conv}(x)
$$

$$
z_{next} = \tilde{z} + g \odot r
$$

This is useful because some regions only need motion, while others need stronger correction for cloud formation or dissipation.

This is often better than an unrestricted residual.

**Best refine block for your 18-to-18 forecasting**

I would use this:

1. Predict flow from current temporal state
2. Warp previous latent
3. Concatenate:
   - warped latent
   - current hidden state
   - future-step embedding
4. Pass through a 3-layer residual CNN
5. Predict:
   - residual correction
   - optional gate
6. Compute:
   $$
   z_{t+k} = \tilde{z}_{t+k} + g_{t+k} \odot r_{t+k}
   $$

That gives you:
- warp for advection
- residual for deformation
- gate for stability

**Loss design for refine block**

You do not usually supervise the refine block directly. You supervise its output through the forecast loss.

Typical total loss:
$$
\mathcal{L} =
\mathcal{L}_{pred}
+ 0.2\mathcal{L}_{warp}
+ 0.05\mathcal{L}_{smooth}
$$

where:
- $\mathcal{L}_{pred}$ supervises the refined and decoded output
- $\mathcal{L}_{warp}$ supervises the pure warped prediction before refinement
- $\mathcal{L}_{smooth}$ regularizes the flow

This is important. If you only supervise the final refined output, the refine block may compensate for poor warp. The separate warp loss keeps the motion path honest.

**Minimal recommended block**

If you want one exact design to start with:

- Input: `concat(warped_latent, hidden_state)`  
- `Conv3x3(2D -> D)`
- `GroupNorm`
- `GELU`
- `Conv3x3(D -> D)`
- `GroupNorm`
- `GELU`
- `Conv3x3(D -> D)` gives residual
- Optional `Conv3x3(D -> D)` + sigmoid gives gate
- Output:
  $$
  z_{next} = \tilde{z} + g \odot r
  $$

Use:
- latent channels $D = 256$ or $512$
- `GroupNorm` if batch size is small
- `GELU` or `SiLU`
- no pooling, no upsampling

**What not to do**

Avoid these early:
- very deep refine blocks
- transformer-only refine blocks
- decoding to image and re-encoding every future step
- predicting the whole next frame only from refine block without residual connection

Those usually make the model harder to train and weaken the motion bias.

If you want, I can give you either:
1. a PyTorch implementation of the refine block,
2. a full 18-step forward pass pseudocode with warp + refine + decode, or
3. a diagram showing where the refine block sits in the architecture.


# how U-net decoder works in this design?

In this design, the U-Net decoder turns each predicted future latent state back into a spatial forecast map. It does not do the temporal forecasting itself. The temporal module plus warp plus refine block produce a future latent for each step, and the decoder reconstructs the corresponding future cloud or irradiance image.

So the role split is:

- Encoder: compress each observed frame into multi-scale spatial features
- Temporal module: learn motion over time
- Warp: move the cloud field
- Refine block: correct motion errors and cloud evolution
- U-Net decoder: reconstruct the future image for each step

For one future step $t+k$, the pipeline is:

$$
z_{t+k} \xrightarrow{\text{decoder}} \hat{I}_{t+k}
$$

where $z_{t+k}$ is the refined latent feature map at low resolution, for example $16 \times 16$ or $32 \times 32$, and $\hat{I}_{t+k}$ is the predicted full-resolution map, for example $128 \times 128$ or $256 \times 256$.

**How the decoder works structurally**

A U-Net decoder is just a sequence of upsampling stages. Each stage:

1. Upsamples the current feature map
2. Merges it with a skip feature from the encoder
3. Applies a few convolutions to recover spatial detail

If the encoder had 4 downsampling levels, the decoder usually has 4 matching upsampling levels.

Example with input image size $256 \times 256$:

- Encoder outputs:
  - skip1: $64 \times 256 \times 256$
  - skip2: $128 \times 128 \times 128$
  - skip3: $256 \times 64 \times 64$
  - skip4: $512 \times 32 \times 32$
  - latent: $512 \times 16 \times 16$

For future step $k$, your temporal model predicts:
- $z_{t+k} \in \mathbb{R}^{B \times 512 \times 16 \times 16}$

Then the decoder does:

1. Upsample $16 \to 32$
2. Concatenate with skip4
3. Convolution block

4. Upsample $32 \to 64$
5. Concatenate with skip3
6. Convolution block

7. Upsample $64 \to 128$
8. Concatenate with skip2
9. Convolution block

10. Upsample $128 \to 256$
11. Concatenate with skip1
12. Convolution block

13. Final $1 \times 1$ conv to output target channels

That final output could be:
- 1 channel for cloud probability
- 1 channel for clear-sky index map
- multiple channels for future satellite bands

**Why skip connections matter here**

The future latent contains motion-aware high-level structure, but low-resolution latents lose fine boundaries. Skip connections inject high-resolution spatial detail back into the decoder.

Without skips, the output tends to be blurry.

In a cloud forecasting setting, skip connections help preserve:
- cloud edges
- texture
- coastlines or terrain context
- local structures useful for irradiance impact

**Important design question: which skip features do you use for future prediction?**

This matters because you are forecasting future frames, not reconstructing the current frame.

There are three common choices.

**Option 1: Use skips from the last observed frame**
This is the simplest and most common starting point.

For all future steps, decode with skip features extracted from the last input frame $I_t$.

So for each future step:
$$
\hat{I}_{t+k} = \mathrm{Decoder}(z_{t+k}, s_t^1, s_t^2, s_t^3, s_t^4)
$$

This works reasonably well because:
- static background stays similar
- recent cloud boundaries still provide useful spatial context

But it can also leak “current appearance bias” into far-horizon forecasts.

**Option 2: Use no skips, only latent decoding**
This is cleaner conceptually but often blurrier. It forces the model to encode everything into the future latent.

Good for a baseline, but usually weaker visually.

**Option 3: Predict future-aware skips**
The temporal model also predicts future multi-scale features, not just one bottleneck latent.

This is more accurate but more complex. Usually not the first model to build.

For your case, I would start with Option 1, then test whether it causes ghosting or over-copying from the last frame.

**What a decoder block looks like**

A standard decoder block is:

1. Bilinear upsample or transposed conv
2. Concatenate with skip
3. `3x3 conv`
4. normalization
5. activation
6. `3x3 conv`
7. normalization
8. activation

Example:

```text
x -> upsample
x -> concat(skip)
x -> conv3x3
x -> groupnorm
x -> GELU
x -> conv3x3
x -> groupnorm
x -> GELU
```

For small batch sizes, use `GroupNorm`, not `BatchNorm`.

Bilinear upsampling plus conv is usually more stable than transposed convolution.

**How it fits into the 18-step rollout**

You do not run one decoder once. You run the same decoder 18 times, once per future latent.

So:

$$
z_{t+1}, z_{t+2}, \dots, z_{t+18}
$$

become

$$
\hat{I}_{t+1}, \hat{I}_{t+2}, \dots, \hat{I}_{t+18}
$$

using the same decoder weights.

That means the decoder is shared across horizons.

**Concrete dataflow**

For each forecast step $k$:

1. Previous latent or hidden state gives flow
2. Warp produces $\tilde{z}_{t+k}$
3. Refine block gives $z_{t+k}$
4. Decoder maps $z_{t+k}$ to $\hat{I}_{t+k}$

Mathematically:

$$
\tilde{z}_{t+k} = \mathrm{Warp}(z_{t+k-1}, F_{t+k-1})
$$

$$
z_{t+k} = \tilde{z}_{t+k} + \mathrm{RefineBlock}(\tilde{z}_{t+k}, h_{t+k-1})
$$

$$
\hat{I}_{t+k} = \mathrm{UNetDecoder}(z_{t+k}, s_t)
$$

where $s_t$ denotes encoder skip features from the last observed frame.

**What should the decoder output?**

Best choices:
- future cloud mask
- future cloud optical proxy
- future clear-sky index map

If your end goal is solar forecasting, predicting a clear-sky index map is often more useful than predicting raw satellite brightness, because it ties more directly to irradiance.

Then a site head can pool around the plant and predict:
$$
\hat{y}_{t+1:t+18}
$$

**Best practical decoder design**

For a first implementation:

- latent channels: 256 or 512
- 4 decoder stages
- bilinear upsampling
- skip concatenation from last observed frame
- 2 conv layers per stage
- GroupNorm + GELU
- final `1x1` conv to output channels

Example channel path:
- latent: 512
- stage 1: 512 + 512 -> 256
- stage 2: 256 + 256 -> 128
- stage 3: 128 + 128 -> 64
- stage 4: 64 + 64 -> 32
- output head: 32 -> target channels

**Potential problem to watch for**

If you use last-frame skip features too aggressively, the decoder may copy current cloud texture into future predictions. That can hurt longer horizons.

Two ways to reduce that:
1. use weaker skips, for example only deeper skips
2. gate the skip contribution

A gated skip version is:

$$
x = \mathrm{concat}(\mathrm{upsample}(x),\ g \odot s)
$$

where $g$ is a learned gate between 0 and 1. This prevents the decoder from blindly copying encoder detail.

For your first model, though, plain skip connections are fine.

**Short version**

The U-Net decoder in this design is a spatial reconstruction module:
- input: one predicted future latent state
- process: repeated upsample + skip fusion + conv refinement
- output: one future forecast map

You apply that same decoder to each of the 18 predicted future latents.

If you want, I can next give you either:
1. a layer-by-layer 18-step architecture with exact tensor shapes, or
2. PyTorch pseudocode for `Encoder`, `RefineBlock`, and `UNetDecoder` together.


