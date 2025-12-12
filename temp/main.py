import asyncio

async def hello(str=''):
  print('Hello world',str)
  

async def main():
    asyncio.create_task(hello('1'))
    print("Hey")
    print("hiii")
    await hello('2')
    print('heyy')
    

print('mike check1')
asyncio.run(main())
print('mike check2')