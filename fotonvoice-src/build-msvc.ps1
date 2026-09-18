$ws = "C:\Users\lap1user\ocdev\fotonvoice-engine"
Start-Process -FilePath "cmd.exe" -ArgumentList "/d","/c","`"$ws\fotonvoice-src\build-msvc.cmd`"" -RedirectStandardOutput "$ws\vc-build-msvc-out.log" -RedirectStandardError "$ws\vc-build-msvc-err.log" -WindowStyle Hidden
