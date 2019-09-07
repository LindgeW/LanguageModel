import os
import json
import glob


class JsonUtil(object):
    def __init__(self):
        super(JsonUtil, self).__init__()

    # 加载单个json文件
    @classmethod
    def load_file(cls, path):
        assert os.path.exists(path)

        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fin:
                # json.loads()  # json字符串 -> python对象
                json_obj = json.load(fin)

        except json.decoder.JSONDecodeError as jde_msg:
            print(jde_msg)
            return None

        return json_obj

    # 加载json文件
    @classmethod
    def file_loader(cls, path):
        assert os.path.exists(path)

        if os.path.isdir(path):
            items = []
            for fname in glob.glob(os.path.join(path, "*.json")):
                json_obj = cls.load_file(fname)
                if isinstance(json_obj, list):
                    items.extend(json_obj)
                else:
                    items.append(json_obj)

            return items
        else:
            return cls.load_file(path)

    # 保存至json文件
    @classmethod
    def dump_json(cls, obj, path):
        with open(path, 'w', encoding='utf-8') as fw:
            # json.dumps()  # python对象 -> json字符串
            json.dump(obj, fw)


class Stu(object):
    def __init__(self, sid, sname):
        self.sid = sid
        self.sname = sname


if __name__ == '__main__':
    # dic = dict([("ads", 1243), ("daew", 545)])
    # print(json.dumps(dic))
    # s = '{"张三": 13, "李四": 434}'
    # print(json.loads(s))

    jo = JsonUtil.file_loader("data/item.json")
    print(jo)

    if isinstance(jo, list):
        for i, j in enumerate(jo):
            # j['raw'] = 'xxx'     # 改变字段
            # j['extra'] = '中国人'  # 添加字段
            print(j.keys())

    JsonUtil.dump_json(jo, "data/stu.json")



