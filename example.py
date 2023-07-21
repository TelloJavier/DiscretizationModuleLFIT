from discretization_module import DiscretizationModule
import pandas as pd
import pylfit

# Set continous features
cont_feats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Import CSV
df = pd.read_csv('wine.data', header=None)

# Split features dataframe and targets dataframe
data = df.drop(0, axis=1)
target = pd.DataFrame(df[0])

######################## KMEANS EXAMPLE ##########################
print()
print('KMEANS example')

# Set clusters list
clusters_list = 13 * [3]

# Build module
dm = DiscretizationModule('kmeans')

# Permform the discretization
tuples = dm.fit_transform_and_build_tuples(data, cont_feats, target, clusters_list)

# Build LFIT model and print summary
feature_names = ['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315_of_diluted_wines','Proline']
target_name = ['Class']

dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=tuples, feature_names=feature_names, target_names=target_name)

print("Initialize a DMVLP with the dataset variables and set PRIDE as learning algorithm")
model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
model.compile(algorithm="pride")
model.summary()

######################## CAIM EXAMPLE ##########################
print()
print('CAIM example')

# Reset features and targets dataframes
data = df.drop(0, axis=1)
target = pd.DataFrame(df[0])

# Build module
dm = DiscretizationModule('caim')

# Permform the discretization
tuples = dm.fit_transform_and_build_tuples(data, cont_feats, target)

# Build LFIT model and print summary
feature_names = ['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315_of_diluted_wines','Proline']
target_name = ['Class']

dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=tuples, feature_names=feature_names, target_names=target_name)

print("Initialize a DMVLP with the dataset variables and set PRIDE as learning algorithm")
model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
model.compile(algorithm="pride")
model.summary()