; Inno Setup script for the Windows installer (#44), built by scripts/build_installer.py from the PyInstaller folder:
;   ISCC /DSourceDir=<dist\omniscan> /DOutputDir=<dist> /DOutputName=<file name> /DAppVersion=<x.y.z> /DIconFile=<ico>
; Installs for the current user (no administrator rights) under %LOCALAPPDATA%\Programs\OmniScan, with a Start menu
; entry, an optional desktop icon and an uninstaller. Unsigned until a code-signing certificate exists
; (docs/OPEN_QUESTIONS.md B8): Windows SmartScreen asks once before the first run.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{7B0F3E2A-4C1D-4E8B-9A57-0B4D6E2F9C31}
AppName=OmniScan
AppVersion={#AppVersion}
AppPublisher=OmniScan
AppPublisherURL=https://github.com/Nawid3333/OmniScan
DefaultDirName={localappdata}\Programs\OmniScan
DefaultGroupName=OmniScan
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\OmniScan.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\OmniScan"; Filename: "{app}\OmniScan.exe"
Name: "{autodesktop}\OmniScan"; Filename: "{app}\OmniScan.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\OmniScan.exe"; Description: "{cm:LaunchProgram,OmniScan}"; Flags: nowait postinstall skipifsilent
