// Deep-Rep Standalone Windows Launcher
// Original Deep-Rep Framework Copyright (c) 2021 Huawei Technologies Co., Ltd.
// Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)
// https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
// Academic / non-commercial research use only.

using System;
using System.IO;
using System.Diagnostics;
using System.Windows.Forms;

namespace DeepRep
{
    class Program
    {
        [STAThread]
        static int Main(string[] args)
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string scriptPath = Path.Combine(baseDir, "deeprep_gui.py");
                
                if (!File.Exists(scriptPath))
                {
                    MessageBox.Show(
                        "Cannot find deeprep_gui.py in application directory:\n" + baseDir,
                        "Deep-Rep Launcher Error",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return 1;
                }

                // Check Python candidates
                string pythonExe = FindPython();
                if (string.IsNullOrEmpty(pythonExe))
                {
                    DialogResult res = MessageBox.Show(
                        "No Python installation detected.\n\nWould you like to run the installer (DeepRepInstaller.exe)?",
                        "Deep-Rep - Python Not Found",
                        MessageBoxButtons.YesNo,
                        MessageBoxIcon.Warning
                    );
                    if (res == DialogResult.Yes)
                    {
                        string installer = Path.Combine(baseDir, "install_windows.bat");
                        if (File.Exists(installer))
                        {
                            Process.Start(installer);
                        }
                    }
                    return 1;
                }

                // Build argument string
                string arguments = "\"" + scriptPath + "\"";
                if (args != null && args.Length > 0)
                {
                    foreach (string arg in args)
                    {
                        arguments += " \"" + arg + "\"";
                    }
                }

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonExe;
                psi.Arguments = arguments;
                psi.WorkingDirectory = baseDir;
                psi.UseShellExecute = false;

                Process.Start(psi);
                return 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error launching Deep-Rep:\n" + ex.Message,
                    "Deep-Rep Error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }
        }

        static string FindPython()
        {
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string[] candidates = new string[]
            {
                Path.Combine(localAppData, @"Programs\Python\Python312\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python311\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python310\python.exe"),
                @"C:\Program Files\Python312\python.exe",
                @"C:\Program Files\Python311\python.exe",
                @"C:\Program Files\Python310\python.exe"
            };

            foreach (string p in candidates)
            {
                if (File.Exists(p)) return p;
            }

            // Check PATH
            string pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                foreach (string dir in pathEnv.Split(';'))
                {
                    string trimmed = dir.Trim();
                    if (!string.IsNullOrEmpty(trimmed))
                    {
                        string test = Path.Combine(trimmed, "python.exe");
                        if (File.Exists(test) && !test.ToLower().Contains("windowsapps"))
                        {
                            return test;
                        }
                    }
                }
            }

            return null;
        }
    }
}
