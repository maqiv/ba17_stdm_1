import analysis.data_analysis as da

da.calculate_test_acccuracies("BiLSTM128_l2_150ms_sp630", True, True, True)
da.calculate_test_acccuracies("BiLSTM128_l2_150ms_sp630_best", True, True, True)

da.calculate_test_acccuracies("BiLSTM128_l2_150ms_sp630_ep4000", True, True, True)
da.calculate_test_acccuracies("BiLSTM128_l2_150ms_sp630_ep4000_best", True, True, True)