# VideoSubFinder

Do not commit the extracted VideoSubFinder binaries or DLL files.

Download VideoSubFinder from:

https://sourceforge.net/projects/videosubfinder/

Extract the archive into this directory so the Windows runtime framework can
find:

```text
res/tool/subfinder/VideoSubFinderWXW.exe
res/tool/subfinder/test.srt
```

After extraction, copy:

```text
res/tool/general.clg
```

over:

```text
res/tool/subfinder/settings/general.cfg
```

The Linux Minecraft training pipeline does not use this tool.
