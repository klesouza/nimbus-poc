using Microsoft.ML.Data;

namespace predict.dotnet
{
    public class Dummy
    {
    [LoadColumn(0)] public float f0 {get;set;}
    [LoadColumn(1)] public float f1 {get;set;}
    [LoadColumn(2)] public float f2 {get;set;}
    [LoadColumn(3)] public float f3 {get;set;}
    [LoadColumn(4)] public float f4 {get;set;}
    [LoadColumn(5)] public string c {get;set;}
    }
}