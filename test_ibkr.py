import asyncio
import nest_asyncio

# Fix for Python 3.10+ event loop changes
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()

from ib_insync import IB

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
print('Connected:', ib.isConnected())
print('Accounts:', ib.managedAccounts())
ib.disconnect()