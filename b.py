from time import *
import  requests,re,os
islin=True
surl='http://10.1.100.2:1000/login?a1b2c3d4e5f6g7h8'
a=''
headers={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:59.0) Gecko/20100101 Firefox/59.0","Content-Type":"application/x-www-form-urlencoded"}

def req(m,url,**k):
    for i in range(5):
            try:
                r=requests.request(m,url,headers=headers,**k)
                break
            except Exception as e:
                print(e)
                sleep(5)
    else:ak(str(e))
    return r
def r(a):
    r=req('get',surl)
#    print(r.text)
    print(re.search('"magic" value="((.*?))"',r.text).groups()[1])
    magic=(re.search('"magic" value="((.*?))"',r.text).groups()[1])
    
    print(a[0])
    d={
        'username':a[0],
        'password':a[1],
        'magic':magic
    }
    r=req('post',r.url,data=d)

    t='authentication failed'
    if t in r.text:return(t);
    t='over limit'
    if t in r.text:return(t);

    if 'keepalive' in r.url:
        if islin:
            cnt=0
            # webbrowser.get(bb).open(r.url+'&t='+ctd().replace(' ','/'))
            while 1:
                print(r.url)
                r=requests.get(r.url)
                print(cnt);cnt+=1
                sleep(60)

        else:
            a=bb.format(r.url+'^&t='+ctd().replace(' ','/'))
            # os.system(f'"{a}" &')
        return('Ok')
    if 'expired' in r.text:
        if exp:print(a)
        return('Expired')
    return 'UX' 

print(r(a.split()))    
