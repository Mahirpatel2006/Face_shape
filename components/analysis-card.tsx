"use client"

import { Brain, User, MoveVertical, MoveHorizontal, Orbit, RectangleHorizontal } from "lucide-react"
import { cn } from "@/lib/utils"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface UploadedImage {
  file: File
  preview: string
}

interface AnalysisResult {
  face_shape: string
  face_length_cm: number
  cheekbone_width_cm: number
  jaw_width_cm: number
  forehead_width_cm: number
  jaw_curve_ratio: number
  processed_image: string
}

interface AnalysisCardProps {
  uploadedImageUrl: string
  analysis: AnalysisResult | null
  isAnalyzing: boolean
}

const StatCard = ({ icon: Icon, label, value, delay, tooltipText }: { icon: React.ElementType; label: string; value: string; delay: string; tooltipText?: string }) => {
    const cardContent = (
        <div
            className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-2xl p-4 flex flex-col items-center justify-center text-center shadow-lg animate-fade-in-up transition-all duration-300 hover:bg-white/70 hover:scale-105 h-full"
            style={{ animationDelay: delay }}
        >
            <div className="w-12 h-12 bg-gradient-to-br from-slate-700 to-slate-900 rounded-full flex items-center justify-center mb-3 shadow-inner">
                <Icon className="w-6 h-6 text-white" />
            </div>
            <span className="text-sm font-medium text-slate-600">{label}</span>
            <span className="text-2xl font-bold text-slate-900">{value}</span>
        </div>
    );

    if (tooltipText) {
        return (
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger className="w-full h-full">{cardContent}</TooltipTrigger>
                    <TooltipContent>
                        <p>{tooltipText}</p>
                    </TooltipContent>
                </Tooltip>
            </TooltipProvider>
        )
    }

    return cardContent;
};

export default function AnalysisCard({ uploadedImageUrl, analysis, isAnalyzing }: AnalysisCardProps) {
  const displayImage = analysis?.processed_image
    ? `http://localhost:5000/uploads/${analysis.processed_image}`
    : uploadedImageUrl
  
  
  const faceLengthDisplay = analysis ? (analysis.face_length_cm + 2.7).toFixed(1) + " cm" : "N/A";
  const foreheadWidthDisplay = analysis ? analysis.forehead_width_cm.toFixed(1) + " cm" : "N/A";
  const cheekboneWidthDisplay = analysis ? analysis.cheekbone_width_cm.toFixed(1) + " cm" : "N/A";
  const jawWidthDisplay = analysis ? analysis.jaw_width_cm.toFixed(1) + " cm" : "N/A";
  const jawCurveDisplay = analysis ? `1:${analysis.jaw_curve_ratio.toFixed(2)}` : "N/A";

  return (
    <div className="grid lg:grid-cols-5 gap-8 mb-16 items-start mt-12">
      <div className="lg:col-span-2 relative group animate-fade-in">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-3xl blur-lg opacity-20 group-hover:opacity-40 transition duration-1000" />
        <div className="relative bg-black/70 backdrop-blur-md rounded-3xl overflow-hidden shadow-2xl border-2 border-white/20">
          <div className="relative z-10 p-4">
            <div className="relative aspect-square rounded-2xl overflow-hidden bg-slate-900 shadow-inner">
              <img
                src={displayImage || "/placeholder.svg"}
                alt="Face analysis"
                className="w-full h-full object-cover transition-all duration-500 group-hover:scale-105"
              />
              {isAnalyzing && (
                <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center space-y-4">
                  <div className="w-16 h-16 border-4 border-t-transparent border-blue-400 rounded-full animate-spin"></div>
                  <p className="text-white font-semibold tracking-wider">ANALYZING...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="lg:col-span-3 relative animate-fade-in" style={{ animationDelay: "0.2s" }}>
        <div className="relative bg-white/70 backdrop-blur-xl rounded-3xl overflow-hidden shadow-2xl border border-slate-200 min-h-[480px]">
          <div className="relative z-10 p-8">
            {isAnalyzing ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-8">
                <div className="w-20 h-20 rounded-full bg-slate-200 flex items-center justify-center mb-6">
                  <Brain className="w-10 h-10 text-slate-500 animate-pulse" />
                </div>
                <p className="text-slate-700 text-2xl font-bold mb-2">Performing AI Analysis...</p>
                <p className="text-slate-500 text-lg">Our algorithms are detecting your facial landmarks.</p>
              </div>
            ) : analysis ? (
              <div className="flex flex-col h-full justify-between space-y-6">
                <div className="text-center p-6 bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl shadow-lg animate-scale-in">
                  <h4 className="text-xl text-slate-300 mb-1 animate-fade-in-up" style={{ animationDelay: "0.2s" }}>Predicted Face Shape</h4>
                  <h3 className="text-7xl font-extrabold text-white tracking-tight animate-fade-in-up" style={{ animationDelay: "0.3s" }}>{analysis.face_shape}</h3>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <StatCard icon={MoveVertical} label="Face Length" value={faceLengthDisplay} delay="0.5s" />
                    <StatCard icon={RectangleHorizontal} label="Forehead Width" value={foreheadWidthDisplay} delay="0.6s" />
                    <StatCard icon={MoveHorizontal} label="Cheekbone Width" value={cheekboneWidthDisplay} delay="0.7s" />
                    <StatCard icon={MoveHorizontal} label="Jaw Width" value={jawWidthDisplay} delay="0.8s" />
                    <div className="md:col-span-2">
                        <StatCard
                            icon={Orbit}
                            label="Jaw Sharpness"
                            value={jawCurveDisplay}
                            delay="0.9s"
                            tooltipText="Ratio of face length to jaw curve. A higher value suggests a more defined, angular jawline."
                        />
                    </div>
                </div>
              </div>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-8 animate-fade-in">
                <div className="w-20 h-20 rounded-full bg-slate-200 flex items-center justify-center mb-6">
                  <Brain className="w-10 h-10 text-slate-400" />
                </div>
                <p className="text-slate-700 text-2xl font-bold">Your Analysis Appears Here</p>
                <p className="text-slate-500 text-lg max-w-xs">Upload an image to get started with your personalized face shape analysis.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
