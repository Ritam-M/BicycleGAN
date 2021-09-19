def Train():
    valid = 1
    fake = 0

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # -------------------------------
            #  Train Generator and Encoder
            # -------------------------------

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            # ----------
            # cVAE-GAN
            # ----------

            # Produce output using encoding of B (cVAE-GAN)
            mu, logvar = encoder(real_B)
            encoded_z = reparameterization(mu, logvar)
            fake_B = generator(real_A, encoded_z)

            # Pixelwise loss of translated image by VAE
            loss_pixel = mae_loss(fake_B, real_B)
            # Kullback-Leibler divergence of encoded B
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
            # Adversarial loss
            loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

            # ---------
            # cLR-GAN
            # ---------

            # Produce output using sampled z (cLR-GAN)
            sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim))))
            _fake_B = generator(real_A, sampled_z)
            # cLR Loss: Adversarial loss
            loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

            # ----------------------------------
            # Total Loss (Generator + Encoder)
            # ----------------------------------

            loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl

            loss_GE.backward(retain_graph=True)
            optimizer_E.step()

            # ---------------------
            # Generator Only Loss
            # ---------------------

            # Latent L1 loss
            _mu, _ = encoder(_fake_B)
            loss_latent = opt.lambda_latent * mae_loss(_mu, sampled_z)

            loss_latent.backward()
            optimizer_G.step()

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)
            # ----------------------------------

            optimizer_D_VAE.zero_grad()

            loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)

            loss_D_VAE.backward()
            optimizer_D_VAE.step()

            # ---------------------------------
            #  Train Discriminator (cLR-GAN)
            # ---------------------------------

            optimizer_D_LR.zero_grad()

            loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)

            loss_D_LR.backward()
            optimizer_D_LR.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D_VAE.item(),
                    loss_D_LR.item(),
                    loss_GE.item(),
                    loss_pixel.item(),
                    loss_kl.item(),
                    loss_latent.item(),
                    time_left,
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_LR.state_dict(), "saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, epoch))
