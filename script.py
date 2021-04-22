import codecademylib3_seaborn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#loading the breast_cancer_data from datasets module
breast_cancer_data = datasets.load_breast_cancer()


print(breast_cancer_data.data[0:3])
print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data ,  validation_data ,training_labels , validation_labels = train_test_split(breast_cancer_data.data , breast_cancer_data.target , test_size = 0.2, random_state = 100)

print(len(training_data))
print(len(training_labels))

accuracies = []
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data , training_labels)
  accuracies.append(classifier.score(validation_data , validation_labels))
#predicting the training_label using one of the trainig data (datapoints considering x value)
y_predict = classifier.predict([breast_cancer_data.data[-2]])
print(y_predict)

k_list = range(1, 101)

plt.plot(k_list , accuracies)
plt.title("Graph of the accuracy of breast cancer using k values from 1 to 100")
plt.xlabel('k(n_neighbors)')
plt.ylabel('Validation Accuracy')
plt.show()







