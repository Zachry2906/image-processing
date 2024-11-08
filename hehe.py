# GRADED FUNCTION: create_model
def create_model():
    """Create the classifier model

    Returns:
        tf.keras.model.Sequential: CNN for multi-class classification
    """
    ### START CODE HERE ###      
    
    # Define the model
    # Use no more than 2 Conv2D and 2 MaxPooling2D
    model = tf.keras.models.Sequential([ 
        # Define an input layer
        tf.keras.Input(shape=(28, 28, 1)),
        # Rescale images
        tf.keras.layers.Rescaling(1./255),
        ]) 

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    ### END CODE HERE ### 
    return model