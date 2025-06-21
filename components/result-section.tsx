'use client';

import { Card } from '@/components/ui/card';

interface ResultSectionProps {
  result: {
    face_shape: string;
    face_length: number;
    cheekbone_width: number;
    jaw_width: number;
    forehead_width: number;
    jaw_curve_ratio: number;
    processed_image: string;
  };
}

export function ResultSection({ result }: ResultSectionProps) {
  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">Analysis Results</h2>
          <p className="text-gray-600">Your face shape has been analyzed</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">Face Shape</h3>
              <p className="text-3xl font-bold text-blue-600">{result.face_shape}</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-600">Face Length</h4>
                <p className="text-xl font-semibold">{result.face_length.toFixed(2)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-600">Jaw Curve</h4>
                <p className="text-xl font-semibold">{result.jaw_curve_ratio.toFixed(3)}</p>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">Width Measurements</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Cheekbone Width:</span>
                  <span className="font-semibold">{result.cheekbone_width.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Jaw Width:</span>
                  <span className="font-semibold">{result.jaw_width.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Forehead Width:</span>
                  <span className="font-semibold">{result.forehead_width.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6">
          <img
            src={`/uploads/${result.processed_image}`}
            alt="Processed face"
            className="w-full rounded-lg shadow-lg"
          />
        </div>
      </div>
    </Card>
  );
} 