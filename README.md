# about-d2l.ai
gluoncv &amp; mxnet Error

###############################################
###    cv code from aws accuracy: 0.88~1.0  ###
###    if you want to use as follow:        ###
###############################################  


data setting brose:
https://www.makesense.ai/


```
pip install gluoncv
```


if you want to use gpu training your datasets,there as follow: 
```
pip install --pre --upgrade mxnet-cu100
```

Error here!
then, there some trouble in my computer

```
Traceback (most recent call last):
  File "E:/tuxiangshibie/train.py", line 1, in <module>
    import gluoncv as gcv
  File "C:\Users\yang\AppData\Local\Programs\Python\Python37\lib\site-packages\gluoncv\__init__.py", line 9, in <module>
    import mxnet as mx
  File "C:\Users\yang\AppData\Local\Programs\Python\Python37\lib\site-packages\mxnet\__init__.py", line 24, in <module>
    from .context import Context, current_context, cpu, gpu, cpu_pinned
  File "C:\Users\yang\AppData\Local\Programs\Python\Python37\lib\site-packages\mxnet\context.py", line 24, in <module>
    from .base import classproperty, with_metaclass, _MXClassPropertyMetaClass
  File "C:\Users\yang\AppData\Local\Programs\Python\Python37\lib\site-packages\mxnet\base.py", line 213, in <module>
    _LIB = _load_lib()
  File "C:\Users\yang\AppData\Local\Programs\Python\Python37\lib\site-packages\mxnet\base.py", line 204, in _load_lib
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_LOCAL)
  File "C:\Users\yang\AppData\Local\Programs\Python\Python37\lib\ctypes\__init__.py", line 356, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: [WinError 126] 找不到指定的模块。

```


step 1:
i try to uninstall gluoncv or mxnet to fix my error
but actually not work


step 2:
mybe its my gpu error, so i try this
```
pip install --pre --upgrade mxnet-cu100
```
but same error above.
