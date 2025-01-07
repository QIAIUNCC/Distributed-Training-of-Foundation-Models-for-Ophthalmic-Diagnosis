def get_merged_classes():
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 1,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 1,
                      "DR": 2,
                      "OTHERS": 3,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 2,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 1,
                        "DR": 2,
                        "HR": 3,
                        "CSR": 4
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 1,
                     "DME": 2,
                     "ERM": 3,
                     "RAO": 4,
                     "RVO": 5,
                     "VID": 6
                     }
    uic_dr_classes = {"Control": 0,
                      "Mild": 1,
                      "Moderate": 2,
                      "Severe": 3,
                      }
  

    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
            uic_dr_classes)


def get_full_classes():
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 1,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 1,
                      "DR": 2,
                      "CNV": 3,
                      "OTHERS": 4,
                      "RVO": 5,
                      "CSC": 6,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 2,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 1,
                        "DR": 2,
                        "HR": 3,
                        "CSR": 4
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 1,
                     "DME": 2,
                     "ERM": 3,
                     "RAO": 4,
                     "RVO": 5,
                     "VID": 6
                     }
    uic_dr_classes = {"Control": 0,
                      "Mild": 1,
                      "Moderate": 2,
                      "Severe": 3,
                      }
    
    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
            uic_dr_classes)
