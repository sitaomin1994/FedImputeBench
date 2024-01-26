import pandas as pd
from sklearn.decomposition import PCA
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .utils import (
	normalization, move_target_to_end, convert_gaussian, drop_unique_cols, one_hot_categorical,
)


########################################################################################################################
# Iris
########################################################################################################################
def process_iris(normalize=True, verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.3
	data = pd.read_csv("./data/iris/iris.csv", header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	data['5'] = data['5'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
	target_col = '5'

	if normalize:
		data = normalization(data, target_col)

	# move target to the end of the dataframe
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	# correlation
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1] - 1)],
		'num_cols': data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Wine white
########################################################################################################################
def process_white(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/wine/winequality-white.csv", delimiter=';')
	data = data.dropna()
	# data['target'] = data.apply(lambda row: 0 if row['quality'] <= 5 else 1, axis=1)
	# data = data.drop(['quality'], axis=1)

	target_col = 'quality'
	if normalize:
		data = normalization(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1] - 1)],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Wine red
########################################################################################################################
def process_red(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/wine/winequality-red.csv", delimiter=';')
	data = data.dropna()

	# data['target'] = data.apply(lambda row: 0 if row['quality'] <= 5 else 1, axis=1)
	target_col = 'quality'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Wine all three
########################################################################################################################
def process_wine_three(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	# merge data
	data_white = pd.read_csv("./data/wine/winequality-white.csv", delimiter=';')
	data_white['type'] = 0
	data_red = pd.read_csv("./data/wine/winequality-red.csv", delimiter=';')
	data_red['type'] = 1
	data = pd.concat([data_white, data_red], axis=0)
	data = data.dropna()

	# label
	data['quality'] = data.apply(lambda row: 0 if row['quality'] <= 5 else 1 if row['quality'] <= 7 else 2, axis=1)
	print(data['quality'].value_counts())
	target_col = 'quality'
	data = move_target_to_end(data, target_col)

	# normalize
	if normalize:
		data = normalization(data, target_col, categorical_cols=['type'])

	# correlation
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1] - 1)],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Spambase
########################################################################################################################
def process_spam(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/spambase/spambase.csv", header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '58'

	# Normalization
	if normalize:
		data = normalization(data, target_col)

	# move target to the end of the dataframe
	data = move_target_to_end(data, target_col)

	# split train and test
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	print(important_features)

	data_config = {
		'target': '58',
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Blocks
########################################################################################################################
def process_blocks(normalize=True, verbose=False, threshold=0.2):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/block/page-blocks.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '11'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	correlation_ret = data.corrwith(data['11'], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Ecoli
########################################################################################################################
def process_ecoli(normalize=True, verbose=False, threshold=0.2):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/ecoli/ecoli.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data = data.drop([0], axis=1)
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '8'
	if normalize:
		data = normalization(data, target_col)
	data[target_col] = data[target_col].map(
		{'cp': 0, 'im': 1, 'pp': 2, 'imU': 3, 'om': 4, 'omL': 5, 'imL': 5, 'imS': 5}
	)
	data = data.dropna()
	data = move_target_to_end(data, target_col)
	# data[target_col] = pd.factorize(data[target_col])[0]
	data = drop_unique_cols(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Glass
########################################################################################################################
def process_glass(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/glass/glass.csv", delimiter=',', header=None)
	data = data.dropna()
	data = data.drop([0], axis=1)
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '10'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data = data[data[target_col] != 6]
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Optical Digits
########################################################################################################################
def process_optdigits(normalize=True, verbose=False, threshold=0.5):
	if threshold is None:
		threshold = 0.5
	data_test = pd.read_csv("./data/optdigits/optdigits_test.csv", delimiter=',', header=None)
	data_train = pd.read_csv("./data/optdigits/optdigits_train.csv", delimiter=',', header=None)
	data = pd.concat([data_test, data_train]).reset_index(drop=True)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '65'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Segmentation
########################################################################################################################
def process_segmentation(normalize=True, verbose=False, threshold=0.5):
	if threshold is None:
		threshold = 0.5
	data_test = pd.read_csv("./data/segment/segmentation_test.csv", delimiter=',')
	data_train = pd.read_csv("./data/segment/segmentation.csv", delimiter=',')
	data = pd.concat([data_test, data_train]).reset_index(drop=True)
	data = data.dropna()
	target_col = 'TARGET'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col].value_counts()
	data[target_col] = data[target_col].map(
		{
			'BRICKFACE': 0,
			'SKY': 1,
			'FOLIAGE': 2,
			'CEMENT': 3,
			'WINDOW': 4,
			'PATH': 5,
			'GRASS': 6,
		}
	)
	data = data.drop(["REGION-PIXEL-COUNT"], axis=1)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Sonar
########################################################################################################################
def process_sonar(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/sonar/sonar.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '61'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col] = data[target_col].map(
		{
			'R': 0,
			'M': 1,
		}
	)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Sensor
########################################################################################################################
def process_sensor(normalize=True, verbose=False, threshold=0.2, pca=False):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/sensor/Sensorless_drive_diagnosis.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '49'

	if pca:
		pca = PCA(n_components=10)
		pca.fit(data.drop(target_col, axis=1))
		data = pd.concat([pd.DataFrame(pca.transform(data.drop(target_col, axis=1))), data[target_col]], axis=1)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data = data[important_features + [target_col]]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Waveform
########################################################################################################################
def process_waveform(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/waveform/waveform-5000.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '41'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col] = data[target_col].map({'N': 0, 'P': 1})
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Yeast
########################################################################################################################
def process_yeast(normalize=True, verbose=False, threshold=0.4):
	if threshold is None:
		threshold = 0.4
	data = pd.read_csv("./data/yeast/yeast.csv", delimiter='\s+', header=None)
	data = data.drop([0], axis=1)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '9'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col].value_counts()
	data[target_col] = data[target_col].map(
		{
			'CYT': 0,
			'NUC': 1,
			'MIT': 2,
			'END': 3,
			'ME3': 4,
			'ME2': 5,
			'ME1': 6,
			'EXC': 7,
			'VAC': 8,
			'POX': 9,
			'ERL': 10,
		}
	)
	data = data[data[target_col] != 10]
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Letter
########################################################################################################################
def process_letter(normalize=True, verbose=False, threshold=0.3):
	if threshold is None:
		threshold = 0.3
	data = pd.read_csv("./data/letter/letter-recognition.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '1'
	data = move_target_to_end(data, target_col)
	# convert columns to gaussian
	data = convert_gaussian(data, target_col)
	if normalize:
		data = normalization(data, target_col)
	# data[target_col].value_counts()
	data[target_col] = data[target_col].map(lambda x: ord(x) - ord('A'))
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Raisin
########################################################################################################################
def process_raisin(normalize=True, verbose=False, threshold=0.2):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/raisin/Raisin_Dataset.csv", delimiter=',')
	data = data.dropna()
	target_col = 'Class'
	data = move_target_to_end(data, target_col)
	if normalize:
		data = normalization(data, target_col)
	data[target_col] = data[target_col].map({"Kecimen": 0, "Besni": 1})
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config

########################################################################################################################
# Telugu Vowel
########################################################################################################################
def process_telugu_vowel(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/telugu_vowel/telugu.csv", delimiter=',', header=0)
	data = data.dropna()
	target_col = 'class'

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	important_features = data.columns.tolist()
	important_features.remove(target_col)

	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'image'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Telugu Tabular
########################################################################################################################
def process_telugu_tabular(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/telugu_tabular/telugu.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '1'

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	important_features = data.columns.tolist()
	important_features.remove(target_col)

	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Wine
########################################################################################################################
def process_wine(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/wine2/wine.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '1'

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


def process_wifi(normalize=True, verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.1

	data = pd.read_csv("./data/wifi_localization/wifi_localization.csv", delimiter='\s+', header=None)
	data = data.dropna()
	# data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = 7

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


def process_firewall(normalize=True, verbose=False, threshold=None):
	threshold = 0.1
	if threshold is None:
		threshold = 0.1

	data = pd.read_csv("./data/firewall/log2.csv")
	data = data[data['Action'] != 'reset-both']
	target_col = 'Action'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()
	source_freq = data['Source Port'].value_counts(normalize=True)
	source_freq.name = 'source_freq'
	destination_freq = data['Destination Port'].value_counts(normalize=True)
	destination_freq.name = 'destination_freq'
	nat_source_freq = data['NAT Source Port'].value_counts(normalize=True)
	nat_source_freq.name = 'nat_source_freq'
	nat_destination_freq = data['NAT Destination Port'].value_counts(normalize=True)
	nat_destination_freq.name = 'nat_destination_freq'

	data = data.merge(source_freq, how='left', left_on='Source Port', right_index=True)
	data = data.merge(destination_freq, how='left', left_on='Destination Port', right_index=True)
	data = data.merge(nat_source_freq, how='left', left_on='NAT Source Port', right_index=True)
	data = data.merge(nat_destination_freq, how='left', left_on='NAT Destination Port', right_index=True)
	# data['sd_pair'] = data[['Source Port', 'Destination Port']].apply(lambda x: tuple(x), axis=1)
	# data['nat_sd_pair'] = data[['NAT Source Port', 'NAT Destination Port']].apply(lambda x: tuple(x), axis=1)
	# sd_pair_freq = data['sd_pair'].value_counts(normalize=True)
	# sd_pair_freq.name = 'sd_pair_freq'
	# nat_sd_pair_freq = data['nat_sd_pair'].value_counts(normalize=True)
	# nat_sd_pair_freq.name = 'nat_sd_pair_freq'
	# data = data.merge(sd_pair_freq, how = 'left', left_on='sd_pair', right_index=True)
	# data = data.merge(nat_sd_pair_freq, how = 'left', left_on='nat_sd_pair', right_index=True)

	data = data.drop(
		[
			'Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port', 'nat_source_freq'
		], axis=1
	)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	# # # sample balance
	# # data_y0 = data[data[target_col] == 0]
	# # data_y1 = data[data[target_col] == 1]
	# # data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
	# # data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_dry_bean(normalize=True, verbose=False, threshold=None, guassian=False):
	normalize = True
	if threshold is None:
		threshold = 0.1

	data = pd.read_excel("./data/dry_bean/Dry_Bean_Dataset.xlsx")
	target_col = 'Class'
	# data = data[data['Action'] != 'reset-both']
	# target_col = 'Action'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()
	data = data.drop(['Extent', 'Solidity', 'ShapeFactor4', 'roundness'], axis=1)

	if guassian:
		data = convert_gaussian(data, target_col)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_pendigits(normalize=True, verbose=False, threshold=None, gaussian=False):
	if threshold is None:
		threshold = 0.1

	data_train = pd.read_csv("./data/pendigits/pendigits.tra", sep=',', header=None)
	data_test = pd.read_csv("./data/pendigits/pendigits.tes", sep=',', header=None)
	data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)
	data = data.dropna()
	data.columns = [str(i) for i in range(data.shape[1])]
	target_col = '16'
	data[target_col] = pd.factorize(data[target_col])[0]

	if gaussian:
		data = convert_gaussian(data, target_col)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_avila(normalize=True, verbose=False, threshold=None):

	if threshold is None:
		threshold = 0.1

	data_train = pd.read_csv("./data/avila/avila-tr.txt", sep=',', header=None)
	data_test = pd.read_csv("./data/avila/avila-ts.txt", sep=',', header=None)
	data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)

	data = data.dropna()
	data.columns = [str(i) for i in range(data.shape[1])]
	target_col = '10'
	data = data[data[target_col].isin(['A', 'F', 'E', 'I', 'X', 'H', 'G', 'D'])]
	data[target_col] = pd.factorize(data[target_col])[0]

	# if normalize:
	# 	data = normalization(data, target_col)

	# #data = convert_gaussian(data, target_col)
	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
#
# Regression
#
########################################################################################################################

########################################################################################################################
# Diabetes
########################################################################################################################

