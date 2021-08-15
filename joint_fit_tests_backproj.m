path=fileparts(which('pwlfmd.py'));
if count(py.sys.path,path)==0
    insert(py.sys.path,int32(0),path);
end

MDFit()