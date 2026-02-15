#!/usr/bin/env python3
"""
5 train a transformer
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler
    """
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        return lr



def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    creates and trains a transformer model for machine
    translation of Portuguese to English
    """
    dataset = Dataset(batch_size, max_len)
    transformer = Transformer(N, dm, h, hidden,
                              dataset.tokenizer_pt.vocab_size + 2,
                              dataset.tokenizer_en.vocab_size + 2,
                              max_len, max_len)
    lr_schedule = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    tr_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    tr_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(epochs):
        tr_accuracy.reset_state()
        tr_loss.reset_state()

        for batch, (inp, tar) in enumerate(dataset.data_train):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions = transformer(inp, tar_inp, training=True,
                                          encoder_mask=enc_padding_mask,
                                          look_ahead_mask=combined_mask,
                                          decoder_mask=dec_padding_mask)
                token_losses = loss_object(tar_real, predictions)

            gradients = tape.gradient(token_losses, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            tr_loss.update_state(token_losses)
            tr_accuracy.update_state(tar_real, predictions)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch}: '
                      f'Loss {tr_loss.result():.2f} '
                      f'Accuracy {tr_accuracy.result():.2f} ')

        print(f'Epoch {epoch + 1}: '
              f'Loss {tr_loss.result():.2f} '
              f'Accuracy {tr_accuracy.result():.2f}')

    return transformer
