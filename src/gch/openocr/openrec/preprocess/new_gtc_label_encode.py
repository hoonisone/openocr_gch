from openocr.openrec.preprocess import dynamic_import


class NewGTCLabelEncode:
    """GCH: SMTR (or other) + CTC heads with explicit ``ctc_label_encode`` in config.

    Merges into one sample dict (keeps ``image`` etc.) and sets ``gtc_label`` /
    ``ctc_label`` for :class:`GTCLoss` / :class:`GTCDecoder` dict batches.
    """

    def __init__(
        self,
        gtc_label_encode,
        ctc_label_encode,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        **kwargs,
    ):
        self.gtc_label_encode = dynamic_import(gtc_label_encode['name'])(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            **gtc_label_encode,
        )
        self.ctc_label_encode = dynamic_import(ctc_label_encode['name'])(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            **ctc_label_encode,
        )

    def __call__(self, data):
        gtc_data = self.gtc_label_encode(data.copy())
        if gtc_data is None:
            return None
        ctc_data = self.ctc_label_encode(data.copy())
        if ctc_data is None:
            return None
        # out = gtc_data
        return {
            'gtc_label': gtc_data,
            'ctc_label': ctc_data,
        }


    @property
    def character_str(self):
        return self.gtc_label_encode.character_str
