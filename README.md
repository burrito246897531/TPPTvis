If encountering BadWindow error on Linux, run this command before qtuitest.py:
```
export QT_QPA_PLATFORM=xcb
```
This command temporarily forces the proper window backend used by QT.
