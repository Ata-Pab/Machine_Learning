Remove Turkish characters: "ü", "ö", "ş", "ı", "ğ"

1) Get paths of anchor_images and positive_images from dataset directory and keep total image number
2) Get anchor_dataset and positive_dataset as tensors using tf.data.Dataset.from_tensor_slices() and taken paths
3) Shuffle anchor_images and positive_images
4) Set negative_images as anchor_images + positive_images
5) Shuffle negative_images
6) Get negative_dataset as tensors using tf.data.Dataset.from_tensor_slices, then shuffle
7) Shuffle negative_dataset
8) Assemble anchor_dataset, positive_dataset, negative_dataset and create dataset
9) Shuffle dataset
10) Convert dataset as anchor, positive and negative image sets using preprocess_image() func. => read jpeg files, convert float32 and resize
11) Divide dataset into train_dataset and val_dataset with 0.8 ratio
12) Create batches for train_dataset and val_dataset using train_dataset.batch(32)
13) Create Siamese Neural Network using Transfer Learning  (ResNet50)
14) Create DistanceLayer class in order to compute distance and set this layer as the last layer of the network
    ap_distance = ‖f(A) - f(P)‖², an_distance = ‖f(A) - f(N)‖²
15) Create inputs as keras.layers.Input with names anchor_input, positive_input, and negative_input for feeding the model
16) Give this inputs to the Resnet + custom Siamese model and get the DistanceLayer results as distances 
    
    distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

17) Create SiameseModel class



NOTES:

train_dataset.prefetch(8): Cogu veri kumesi giris ardisik duzeni, onceden getirme cagrisiyla sona ermelidir. Bu, mevcut eleman islenirken sonraki elemanlarin hazirlanmasina izin verir. Bu, onceden getirilen ogeleri depolamak icin ek bellek kullanma pahasina genellikle gecikmeyi ve verimi artirir.