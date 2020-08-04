using System;
using Microsoft.ML;
using System.IO;
using System.Reflection;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Runtime;

namespace predict.dotnet
{
    class Program
    {
        static void Main(string[] args)
        {
            Load();
        }
        public static void Load()
        {
            DataViewSchema schema;
            var ctx = new MLContext();
            var model = ctx.Model.Load(File.OpenRead("../train-model/lgbm_nimbus.zip"), out schema);

            IDataView testData = ctx.Data.LoadFromTextFile<Dummy>(
                "../train-model/dummy_test.csv",
                hasHeader: true,
                separatorChar: ',');
            var keys = new VBuffer<ReadOnlyMemory<char>>();
            schema.GetColumnOrNull("c").Value.GetKeyValues(ref keys);
            var test = ctx.Transforms.Conversion.MapValueToKey("c", "c",
                    addKeyValueAnnotationsAsText: true,
                    keyData: ctx.Data.LoadFromEnumerable(keys.GetValues().ToArray().Select(x => new { Key = x })))
            .Fit(testData).Transform(testData);
            // var tr = model.GetRowToRowMapper(testData.Schema);
            // var predictions = ctx.Model.CreatePredictionEngine<Fares, Prediction>(model);
            var p = test.Preview(10);
            var p2 = testData.Preview(10);
            var preds = model.Transform(test).GetColumn<float>("Score").ToArray();
            Console.WriteLine(string.Join('\n', preds));
        }
    }
}
