from src.modules.data_prep.data_prep_his import (
    process_NHIS_income, process_heart, process_codrna, process_skin, process_codon, process_sepsis,
    process_diabetic, process_diabetic2, process_cardio, process_mimiciii_mortality, process_genetic,
    process_mimiciii_mo2, process_mimic_icd, process_mimic_icd2, process_mimic_mo, process_mimic_los,
    process_breast, process_dermatology, process_pima_diabetes
)

from src.modules.data_prep.data_prep_large import (
    process_ijcnn, process_susy, process_statlog, process_higgs, process_svm
)

from src.modules.data_prep.data_prep_fairness import (
    process_adult, process_default_credit, process_bank_market
)

from src.modules.data_prep.data_prep import (
    process_iris, process_ecoli, process_white, process_red, process_wine_three, process_spam, process_blocks,
    process_glass, process_optdigits, process_segmentation, process_sonar, process_sensor, process_waveform,
    process_yeast, process_letter, process_raisin, process_telugu_vowel, process_telugu_tabular, process_wine,
    process_wifi, process_firewall, process_dry_bean, process_avila, process_pendigits
)

from src.modules.data_prep.data_prep_reg import (
    process_diabetes, process_california_housing, process_housing, process_red_reg, process_white_reg
)


def load_data(dataset_name, normalize=True, verbose=False, threshold=None):

    # ##################################################################################################################
    # Classification
    # ##################################################################################################################
    if dataset_name == 'iris':
        return process_iris(normalize, verbose, threshold)
    elif dataset_name == 'breast':
        return process_breast(normalize, verbose, threshold)
    elif dataset_name == 'ecoli':
        return process_ecoli(normalize, verbose, threshold)
    elif dataset_name == 'white':
        return process_white(normalize, verbose, threshold)
    elif dataset_name == 'red':
        return process_red(normalize, verbose, threshold)
    elif dataset_name == 'wine_quality_all':
        return process_wine_three(normalize, verbose, threshold)
    elif dataset_name == 'spam':
        return process_spam(normalize, verbose, threshold)
    elif dataset_name == 'blocks':
        return process_blocks(normalize, verbose, threshold)
    elif dataset_name == 'glass':
        return process_glass(normalize, verbose, threshold)
    elif dataset_name == 'optdigits':
        return process_optdigits(normalize, verbose, threshold)
    elif dataset_name == 'segmentation':
        return process_segmentation(normalize, verbose, threshold)
    elif dataset_name == 'sonar':
        return process_sonar(normalize, verbose, threshold)
    elif dataset_name == 'sensor':
        return process_sensor(normalize, verbose, threshold)
    elif dataset_name == 'sensor_pca':
        return process_sensor(normalize, verbose, threshold, pca=True)
    elif dataset_name == 'waveform':
        return process_waveform(normalize, verbose, threshold)
    elif dataset_name == 'yeast':
        return process_yeast(normalize, verbose, threshold)
    elif dataset_name == 'letter':
        return process_letter(normalize, verbose, threshold)
    elif dataset_name == 'raisin':
        return process_raisin(normalize, verbose, threshold)
    elif dataset_name == 'telugu_vowel':
        return process_telugu_vowel(normalize, verbose, threshold)
    elif dataset_name == 'telugu_tabular':
        return process_telugu_tabular(normalize, verbose, threshold)
    elif dataset_name == 'wine':
        return process_wine(normalize, verbose, threshold)
    elif dataset_name == 'wifi':
        return process_wifi(normalize, verbose, threshold)
    elif dataset_name == 'default_credit':
        return process_default_credit(normalize, verbose, threshold)
    elif dataset_name == 'firewall':
        return process_firewall(normalize, verbose, threshold)
    elif dataset_name == 'dry_bean':
        return process_dry_bean(normalize, verbose, threshold)
    elif dataset_name == 'dry_bean_g':
        return process_dry_bean(normalize, verbose, threshold, guassian=True)

    ####################################################################################################################
    # Large Dataset
    ####################################################################################################################
    elif dataset_name == 'adult':
        return process_adult(normalize, verbose, threshold)
    elif dataset_name == 'adult_pca':
        return process_adult(normalize, verbose, threshold, sample=False, pca=True, gaussian=True)
    elif dataset_name == 'adult_balanced':
        return process_adult(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'adult_balanced_pca':
        return process_adult(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
    elif dataset_name == 'bank_marketing':
        return process_bank_market(normalize, verbose, threshold)
    elif dataset_name == 'bank_marketing_balanced':
        return process_bank_market(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'bank_balanced_pca':
        return process_bank_market(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
    elif dataset_name == 'ijcnn':
        return process_ijcnn(normalize, verbose, threshold)
    elif dataset_name == 'ijcnn_balanced':
        return process_ijcnn(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'ijcnn_balanced_pca':
        return process_ijcnn(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
    elif dataset_name == 'svm':
        return process_svm(normalize, verbose, threshold)
    elif dataset_name == 'svm_g':
        return process_svm(normalize, verbose, threshold, gaussian=True)
    elif dataset_name == 'pendigits':
        return process_pendigits(normalize, verbose, threshold)
    elif dataset_name == 'pendigits_g':
        return process_pendigits(normalize, verbose, threshold, gaussian=True)
    elif dataset_name == 'statlog':
        return process_statlog(normalize, verbose, threshold)
    elif dataset_name == 'statlog_pca':
        return process_statlog(normalize, verbose, threshold, pca=True, gaussian=True)
    elif dataset_name == 'avila':
        return process_avila(normalize, verbose, threshold)
    elif dataset_name == 'susy':
        return process_susy(normalize, verbose, threshold)
    elif dataset_name == 'susy_g':
        return process_susy(normalize, verbose, threshold, gaussian=True)
    elif dataset_name == 'higgs':
        return process_higgs(verbose, threshold)

    ####################################################################################################################
    # Healthcare Dataset
    ####################################################################################################################
    elif dataset_name == 'nhis_income':
        return process_NHIS_income(pca=False)
    elif dataset_name == 'nhis_income_pca':
        return process_NHIS_income(pca=True)
    elif dataset_name == 'heart':
        return process_heart(pca=True, sample=False)
    elif dataset_name == 'heart_balanced':
        return process_heart(pca=True, sample=True)
    elif dataset_name == 'skin':
        return process_skin(normalize, verbose, threshold, sample=False)
    elif dataset_name == 'skin_balanced':
        return process_skin(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'codrna':
        return process_codrna(normalize, verbose, threshold, sample=False)
    elif dataset_name == 'codrna_balanced':
        return process_codrna(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'codon':
        return process_codon(verbose, threshold)
    elif dataset_name == 'sepsis':
        return process_sepsis(verbose, threshold)
    elif dataset_name == 'diabetic':
        return process_diabetic(verbose, threshold)
    elif dataset_name == 'diabetic_balanced':
        return process_diabetic(verbose, threshold, sample=True)
    elif dataset_name == 'diabetic2':
        return process_diabetic2(verbose, threshold, sample=True)
    elif dataset_name == 'cardio':
        return process_cardio(verbose, threshold)
    elif dataset_name == 'mimiciii_mo':
        return process_mimiciii_mortality()
    elif dataset_name == 'mimiciii_icd':
        return process_mimic_icd2()
    elif dataset_name == 'mimiciii_mo2':
        return process_mimic_mo()
    elif dataset_name == 'mimiciii_los':
        return process_mimic_los()
    elif dataset_name == 'genetic':
        return process_genetic(sample=False)
    elif dataset_name == 'genetic_balanced':
        return process_genetic(sample=True)
    elif dataset_name == 'dermatology':
        return process_dermatology(normalize, verbose, threshold)
    elif dataset_name == "pima_diabetes":
        return process_pima_diabetes(normalize, verbose, threshold)

    ####################################################################################################################
    # Regression
    ####################################################################################################################
    elif dataset_name == 'diabetes':
        return process_diabetes(normalize, verbose, threshold)
    elif dataset_name == 'california_housing':
        return process_california_housing(normalize, verbose, threshold)
    elif dataset_name == 'housing':
        return process_housing(normalize, verbose, threshold)
    elif dataset_name == 'red_reg':
        return process_red_reg(normalize, verbose, threshold)
    elif dataset_name == 'white_reg':
        return process_white_reg(normalize, verbose, threshold)
    else:
        raise Exception("Unknown dataset name {}".format(dataset_name))
