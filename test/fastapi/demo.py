import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from processor.query_processor.logger import logger

# 创建一个 FastAPI 应用实例
app = FastAPI()

@app.get("/",summary="第一个测试")
#下列函数将被调用当用户使用 GET 方法访问根 URL
async def read_root():
    return {"hello":"world"}


#-----------参数解析----------
# 访问 http://127.0.0.1:8001/items/5?q=somequery
#?：分割路由和查询参数，只写 1 次
# item_id: 路径参数 (自动转为 int)
# q: 查询参数 (可选，默认 None)
@app.get("/items/{item_id}",summary="获取指定参数")
async def read_single_item(item_id:int,q:str|None=None):
    return {"item_id":item_id,"q":q}


# 接收? skip=? & limit = ?
@app.get("/items",summary="分页")
async def read_item_list(skip:int = 0,limit:int = 10):
    return {"skip":skip,"limit":limit}


#----------类型检查和错误提示----------
class Item(BaseModel):
    name : str
    price : float
    is_offer : bool = None

# POST 请求接收 JSON 数据
@app.post("/items/",summary="类型检查")
async def create_item(item:Item):
    # item 已经是验证过的 Item 对象
    # 如果客户端传来的 price 是字符串 "abc"，FastAPI 会自动报错
    return {"name":item.name,"price":item.price,"is_offer":item.is_offer}

#用户输入错误数据
#    ↓
#FastAPI 接收请求
#    ↓
#交给 Pydantic 验证  ← Pydantic 登场！
#    ↓
#Pydantic 发现类型不匹配
#    ↓
#生成详细的错误信息
#    ↓
#FastAPI 把错误信息返回给用户


if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8001端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1" ,# 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8001 # 服务端口
    )


