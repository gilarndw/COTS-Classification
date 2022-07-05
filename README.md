# Deep Learning COTS Image Classification with MobileNet V2: Overview
* Collected Binary COTS Images Dataset from kaggle
* Preprocessed the image data through data augmentation to fit the model
* Used MobileNet V2 for Transfer Learning and gained 96% accuracy

## Tools and Resources
* **Tools**: Google Colab Notebook, MobileNet V2
* **Resources**: https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset

## Data Preparation
* Imported data to Google Colab

![image](https://user-images.githubusercontent.com/60825743/177265881-56dd8dac-9606-416b-8af0-5103e20c1bfe.png)

* Created variables to contain the images and labels then put the images and labels into that variables
* Shuffled the data inside the variables
```python
all_paths = []
all_labels = []

for label in os.listdir(data_dir):
  for image in os.listdir(data_dir+label):
    all_paths.append(data_dir+label+'/'+image)
    all_labels.append(label)

  all_paths, all_labels = shuffle(all_paths, all_labels)
```
* Splitted the data for training and validation
```python
#80% for training and 20% for validation

x_train_paths, x_val_paths, y_train, y_val = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42)
```
![image](https://user-images.githubusercontent.com/60825743/177266553-12533e5c-51fd-4884-8776-112d1e295b25.png)

##Data Augmentation
* Augmented the data by giving filters to enhance the brightness, contrast, and color of the images
```python
#Data Augmentation

def augment_image(image):
    if random.uniform(0,1)>0.5:
        image = np.fliplr(image)
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6,1.4))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6,1.4))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.6,1.4))
    return image
```
![image](https://user-images.githubusercontent.com/60825743/177274067-cdb3d571-c3de-4168-8598-d24e293f6e93.png)

This is some example of the image data after given filters

* Given unique labels so the machine can't identify which is which

```python
unique_labels = os.listdir(data_dir)

def encode_label(labels):
    encoded = []
    for x in labels:
        encoded.append(unique_labels.index(x))
    return np.array(encoded)

def decode_label(labels):
    decoded = []
    for x in labels:
        decoded.append(unique_labels[x])
    return np.array(decoded)

def data_gen(paths, labels, batch_size=12, epochs=3, augment=True):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x+batch_size]
            batch_images = open_images(batch_paths, augment=augment)
            batch_labels = labels[x:x+batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels
```

##Bulding the Model
* Used MobileNet v2 as the base model
* Added GlobalAveragePooling, Dense, Dropout Layer and Activation Layer to the model
* Compiled the model with Adam optimizer

![image](https://user-images.githubusercontent.com/60825743/177275962-86fc3d15-6d45-4da5-bab0-d891d29a0f9f.png)

##Model Training and Validation
* Divided the data into 32 batch 
* Trained the model through 6 epochs

```python
batch_size = 32
steps = int(len(x_train_paths)/batch_size)
epochs = 6
history = model.fit(data_gen(x_train_paths, y_train, batch_size=batch_size, epochs=epochs, augment=True),
                    epochs=epochs, steps_per_epoch=steps)
```

![image](https://user-images.githubusercontent.com/60825743/177277249-9533bb4a-89fb-4d2c-9326-e3578b8e860a.png)

* Validation result on the model said the model has 96% accuracy on predicting the image

![image](https://user-images.githubusercontent.com/60825743/177277892-d609ebed-60d3-40bc-9158-7d9c81b12ba2.png)



