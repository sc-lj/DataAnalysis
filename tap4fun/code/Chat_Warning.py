#coding:utf-8
import requests,json
import time,os
import mimetypes

base_url='https://qyapi.weixin.qq.com/cgi-bin/'
corpsecret='DaahMvdBJe4ffUjk2zrGk2CjcXqvWZ98J9-mFs-ufn4'

# token是企业后台去企业微信的后台获取信息时的重要票据，由corpid和secret产生。所有接口在通信时都需要携带此信息用于验证接口的访问权限.
# 每个应用有独立的secret，所以每个应用的access_token应该分开来获取
def get_token(corpsecret=corpsecret):
    # corpid：每个企业都拥有唯一的corpid。
    # secret是企业应用里面用于保障数据安全的“钥匙”，每一个应用都有一个独立的访问密钥，为了保证数据的安全，secret务必不能泄漏
    tokenUrl=base_url+'gettoken'
    values = {'corpid': 'ww224b6e5e6e3f81e2', 'corpsecret': corpsecret}
    req = requests.post(tokenUrl, params=values)
    data = json.loads(req.text)
    '''返回示例
    {
   "errcode":0，
   "errmsg":""，
   "access_token": "accesstoken000001",
   "expires_in": 7200
    }'''
    expires_in= data['expires_in']#凭证的有效时间
    return data["access_token"]

# 发送文件
def send_msg(msg,corpsecret=corpsecret):
    token=get_token(corpsecret)
    url = base_url + 'message/send'
    # agentid企业应用id
    values = {"touser" : "LuJun" ,
      "msgtype":"text",
      "agentid":"1000002",
      "text":{
        "content": "%s"%msg
      },
      "safe":"0"
      }
    requests.post(url,json=values,params={'access_token':token})



def get_media_stoken(media_path,types):
    media_token_url=base_url+'media/upload'
    token = get_token(corpsecret)
    fr = open(media_path, 'rb')
    contents=fr.read()
    fr.close()

    param = {'access_token':token,'type':types}
    req=requests.post(media_token_url,params=param,files={types:contents})
    data = req.text
    data=json.loads(data)
    return data['media_id'],token

def send_picture(media_path,types):
    media_id,token=get_media_stoken(media_path,types)
    print(media_id)
    url = base_url + 'message/send'
    # "toparty" : "PartyID1|PartyID2",
    # "totag" : "TagID1 | TagID2",
    value={
   "touser" : "LuJun",
   "msgtype" : types,
   "agentid" : 1000002,
   types : {
        "media_id" : media_id
        },
   "safe":0
    }
    requests.post(url, json=value,params={'access_token':token})

if __name__ == '__main__':

    try:
        1 / 0
    except ZeroDivisionError as e:
        send_msg(e.args[0],corpsecret=corpsecret)

    # send_picture('/Users/lj/Desktop/十九大心得.docx','file')
    # path='/Users/lj/Desktop/12.gfg'
    # filename=os.path.basename(path)
    # types=mimetypes.guess_type(filename)[0]
    # if types is not None:
    #     types=types.split('/')[0]
    #     if types=='text':
    #         types='file'
    # else:types='file'
    # print  types




