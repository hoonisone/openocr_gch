from openocr.openrec.preprocess import MODULE_MAPPING, dynamic_import

MODULE_MAPPING['GCHLabelEncode'] = 'gch.openocr.openrec.preprocess.gch_label_encode'





class GCHLabelEncode():
    def __init__(self, c_encoder, g_encoder, **kwargs):

        korean_transformer_kwargs = kwargs.get("korean_transformer", {})
        kwargs.pop("korean_transformer", None)
        self.korean_transformer = KoreanTransfomer(
            **korean_transformer_kwargs
        )

        self.c_encoder = dynamic_import(c_encoder['name'])(**c_encoder)
        self.g_encoder = dynamic_import(g_encoder['name'])(**g_encoder)



        # KoreanTransformer에 들어가는 자소 정보와 g_encoder에 등록된 자소 문자 내용이 일치하는지,
        # 자소 정보가 적어도 유형별로 정확한 개수만큼 들어있는지 체크
        assert all([c in self.g_encoder.character_str for c in self.korean_transformer.initials])
        assert all([c in self.g_encoder.character_str for c in self.korean_transformer.medials])
        assert all([c in self.g_encoder.character_str for c in self.korean_transformer.finals])


    def __call__(self, data):
        # apply c_encoder directly
        label_data = {"label": data["label"]}
        c_label = self.c_encoder(label_data)
        if c_label is None:
            return None
        
        label_data = {"label": self.korean_transformer.c2g(data["label"])}
        g_label = self.g_encoder(label_data)
        if g_label is None:
            return None
        
        data["c_label"] = c_label
        data["g_label"] = g_label
        return data

from typing import List, Tuple, Union, Optional

class KoreanTransfomer:

    START_UNICODE_IDX = ord('가') # 44032
    FINAL_UNICODE_IDX = ord('힣') # 55203

    INITIAL_NUM = 19
    VOWEL_NUM = 21
    FINAL_NUM = 28
    INITIAL_DURATION = ord('까') - ord('가') # 588
    VOWEL_DURATION = ord('개') - ord('가') # 28

    def __init__(self,
                 initial_dict_path=None,
                 medial_dict_path=None,
                 final_dict_path=None,
                 **kwargs):

        self.initials = self._load_dict(initial_dict_path) if initial_dict_path else [chr(ord('가')+i*self.INITIAL_DURATION) for i in range(self.INITIAL_NUM)]
        self.medials = self._load_dict(medial_dict_path) if medial_dict_path else [chr(ord('ㅏ')+i) for i in range(self.VOWEL_NUM)] # 글자 중복으로 '아애야...' 가 아닌 'ㅏㅐㅑ...' 으로 생성
        self.finals = self._load_dict(final_dict_path) if final_dict_path else [chr(ord('으')+i) for i in range(self.FINAL_NUM)]

        assert len(self.initials) == self.INITIAL_NUM
        assert len(self.medials) == self.VOWEL_NUM
        assert len(self.finals) == self.FINAL_NUM

    def _load_dict(self, dict_path:str)->List[str]:
        with open(dict_path, 'r') as f:
            return [line.strip() for line in f]



    def c2g(self, text:str)->List[str]: 

        assert isinstance(text, str|List), "text must be a string or list of strings"


        if len(text) == 0:
            return []
        elif len(text) == 1:
            char = text
            assert isinstance(char, str), "char must be a string"

            if not self.START_UNICODE_IDX <= ord(char) <= self.FINAL_UNICODE_IDX:
                return [char]
            
            
            code = ord(char) - self.START_UNICODE_IDX
            initial_index = code // self.INITIAL_DURATION
            medial_index = (code - initial_index * self.INITIAL_DURATION) // self.VOWEL_DURATION
            final_index = code % self.VOWEL_DURATION

            return [self.initials[initial_index], self.medials[medial_index], self.finals[final_index]]
         
        else:
            return sum([self.c2g(char) for char in text], [])

    def _is_composible(self, initial:str, medial:str, final:str)->bool:
        if initial not in self.initials:
            return False
        if medial not in self.medials:
            return False
        if final not in self.finals:
            return False
        return True

    def _is_character(self, char:str)->bool:
        return isinstance(char, str) and len(char) == 1

    def _g2c(self, initial:str, medial:str, final:str)->Optional[str]:
        assert self._is_character(initial), "initial must be a string"
        assert self._is_character(medial), "medial must be a string"
        assert self._is_character(final), "final must be a string"
 


        if not self._is_composible(initial, medial, final):
            return None
        else:
            idx = self.START_UNICODE_IDX + self.initials.index(initial) * self.INITIAL_DURATION
            idx += self.medials.index(medial) * self.VOWEL_DURATION
            idx += self.finals.index(final)
            return chr(idx)

    def g2c(self, text:str|List[str])->Optional[str]:
        assert isinstance(text, str|List), "text must be a string or list of strings"

        i = 0

        new_text = []
        while True:
            if i + 2 > len(text)-1: # 남은 글자가 3개 미만 이면
                new_text.extend(text[i:]) # 남은 글자 추가하고 종료
                break
            else: # 3개 이상 남아있다면 
                result = self._g2c(text[i], text[i+1], text[i+2]) # 3개글자를 한글 자모로 변환 시도
                if result is None: # 변환에 실패 시 
                    new_text.append(text[i]) # 현재 다루는 첫 글자 (initial)을 변환 없이 그대로 추가하고
                    i += 1 # 다음 스텝
                else: # 변환에 성공 시
                    new_text.append(result) # 변환 결과를 추가하고
                    i += 3 # 3개 글자 패스

        return "".join(new_text)