# Model limitations

The fallback checkpoint is a published seven-class EfficientNet-B0 model for HAM10000 / ISIC 2018-style dermoscopic lesions. It is not a universal skin disease classifier and is not trained by this project author. A class added in the admin UI is metadata only until a model is trained with representative labeled examples for that class.

Do not present scores as calibrated clinical probabilities or use this system to diagnose, rule out disease, or delay care. For credible evaluation, use patient-level splits and independent external clinical-photo validation, ideally with dermatologist-reviewed labels.
