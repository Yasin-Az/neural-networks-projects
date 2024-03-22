import tensorflow as tf

'''
This callback saves model or model weights every epoch then 
later we can use the checkpoint to continue our training again
from where it was saved
'''

# set check point into path we want
checkpoint_path = "path/checkpoint.ckpt" # path we want to save our checkpoint, ".ckpt is its foramt"

# create a model checkpoint that saves model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True, # saves only best epochs, if False it saves every epochs
                                                         save_freq="epoch", # saves every epoch
                                                         verbose=1)

# fitting model and using our callback in it
our_model.fit(# .....
          # .....
          # .....
          callbacks=[checkpoint_callback]) # passing our checkpoint

#********************************************************************

# loading checkpoints weights we just saved "load_weights()"
# with this we can return our model to a specific checkpoint

our_model.load_weights(checkpoint_path)

# now our model is ready!