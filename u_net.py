def u_net():
    # optimizer =
    # loss_metric = 
    # metrics = 
    lr = 1e-3
    size = (3, 3)
    myinputs = Input((IMG_WIDTH, IMG_HEIGHT, z))
    conv1 = Conv2D(32, size, activation='relu', padding='same')(myinputs)
    conv1 = Conv2D(32, size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, size, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)
    
    conv3 = Conv2D(128, size, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.3)(pool3)
    
    conv4 = Conv2D(256, size, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, size, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.3)(pool4)
    
    conv5 = Conv2D(512, size, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, size, activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, size, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, size, activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, size, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, size, activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, size, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, size, activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, size, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, size, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(OUTPUT_CHANNELS, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[myinputs], outputs=conv10)

    model.compile(optimizer=optimizer(lr=lr), loss=loss_metric, metrics=metrics)
    
    return model
