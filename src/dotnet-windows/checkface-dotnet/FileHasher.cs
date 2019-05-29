using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace checkface_dotnet
{
    public class FileHasher
    {
        public FileHasher(HashAlgorithmName algorithmName)
        {
            this.AlgorithmName = algorithmName;
        }

        public HashAlgorithmName AlgorithmName { get; }

        public string HashFile(string fileName)
        {
            using(var alg = HashAlgorithm.Create(AlgorithmName.Name))
            {
                using(var fs = new FileStream(fileName, FileMode.Open))
                {
                    var bytes = alg.ComputeHash(fs);

                    // Convert byte array to a string   
                    var builder = new StringBuilder();
                    for (int i = 0; i < bytes.Length; i++)
                    {
                        builder.Append(bytes[i].ToString("X2"));
                    }
                    return builder.ToString();
                }
            }
        }

        public static HashAlgorithmName[] Algorithms = new[]
        {
            HashAlgorithmName.SHA256,
            HashAlgorithmName.MD5,
            HashAlgorithmName.SHA1,
            HashAlgorithmName.SHA512
        };
    }
}
