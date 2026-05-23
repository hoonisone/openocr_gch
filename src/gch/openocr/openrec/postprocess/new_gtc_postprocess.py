from openocr.openrec.postprocess import build_post_process


class NewGTCLabelDecode(object):
    """GCH: separate ``gtc_label_decode`` / ``ctc_label_decode`` configs + dict ``batch``."""

    def __init__(
        self,
        gtc_label_decode=None,
        ctc_label_decode=None,
        character_dict_path=None,
        use_space_char=True,
        only_gtc=False,
        with_ratio=False,
        **kwargs,
    ):
        gtc_label_decode['character_dict_path'] = character_dict_path
        gtc_label_decode['use_space_char'] = use_space_char

        ctc_label_decode['character_dict_path'] = character_dict_path
        ctc_label_decode['use_space_char'] = use_space_char

        self.gtc_label_decode = build_post_process(gtc_label_decode)
        self.ctc_label_decode = build_post_process(ctc_label_decode)

        self.gtc_character = self.gtc_label_decode.character
        self.ctc_character = self.ctc_label_decode.character
        self.only_gtc = only_gtc
        self.with_ratio = with_ratio

    def get_character_num(self):
        return {
            'gtc_num': len(self.gtc_character),
            'ctc_num': len(self.ctc_character),
        }

    def __call__(self, preds, batch=None, *args, **kwargs):
        if self.with_ratio:
            assert isinstance(batch, list), (
                'with_ratio: batch dict path not implemented for NewGTCLabelDecode')
            batch = batch[:-1]

        if isinstance(batch, dict):
            gtc_batch = batch['gtc_label']
        else:
            gtc_batch = batch[:-2] if batch is not None else None

        gtc = self.gtc_label_decode(preds['gtc_pred'],
                                    gtc_batch if batch is not None else None)
        if self.only_gtc:
            return {'gtc_pred': gtc}

        if isinstance(batch, dict):
            ctc_batch = batch['ctc_label']
        else:
            ctc_batch = [None] + batch[-2:] if batch is not None else None
        ctc = self.ctc_label_decode(preds['ctc_pred'],
                                    ctc_batch if batch is not None else None)

        return {'gtc_pred': gtc, 'ctc_pred': ctc, 'pred': ctc}
