from pysnmp.hlapi import *

# Cấu hình thông tin SNMP
community = CommunityData('public', mpModel=0)
target = UdpTransportTarget(('192.168.1.1', 161))

# Thực hiện truy vấn SNMP
errorIndication, errorStatus, errorIndex, varBinds = next(
    getCmd(SnmpEngine(),
           community,
           target,
           ContextData(),
           ObjectType(ObjectIdentity('SNMPv2-MIB', 'sysDescr', 0)))
)

# Xử lý kết quả truy vấn
if errorIndication:
    print('Lỗi: %s' % errorIndication)
elif errorStatus:
    print('Lỗi: %s at %s' % (errorStatus.prettyPrint(), errorIndex and varBinds[int(errorIndex)-1][0] or '?'))
else:
    for varBind in varBinds:
        print('Thông tin trạng thái:', varBind)

