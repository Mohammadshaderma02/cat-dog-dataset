# cat & dog dataset

* To Show Code And More details  <a href='https://colab.research.google.com/drive/1nguOEexRE0bJ-hWMIpIdRHZ9fYPK73Bt?usp=sharing'>Go to Colab</a>

* To Download Dataset <a href='https://www.kaggle.com/datasets/tongpython/cat-and-dog'>Go to Kaggle</a>

## **About cat & dog dataset**
---


*   The dataset provide 25000 cats and dogs images to classify.

---



# > **Unzipping files**
```python
import zipfile

# returns 'Select files'
zip_files = ['test1', 'train']

for zip_file in zip_files:
    with zipfile.ZipFile("/content/drive/MyDrive/dogs-vs-cats.zip".format(zip_file),"r") as z:
        z.extractall(".")
        print("{} unzipped".format(zip_file))

```



 > **Build Model**

 <img src='https://media.geeksforgeeks.org/wp-content/uploads/cat-vs-dog.jpg'   />

>Library
* pandas
* numpy
* os
* keras
* tensorflow
