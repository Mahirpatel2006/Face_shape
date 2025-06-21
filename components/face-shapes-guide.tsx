"use client"

import { User } from "lucide-react"

const faceShapes = [
  {
    shape: "Oval",
    characteristics: ["Balanced proportions", "Gentle curves", "Harmonious features", "Natural symmetry"],
    description:
      "The oval face represents perfect facial harmony with balanced proportions that create a naturally pleasing aesthetic.",
    color: "from-blue-500 to-cyan-500",
  },
  {
    shape: "Round",
    characteristics: ["Soft curves", "Full cheeks", "Gentle jawline", "Youthful appearance"],
    description:
      "Round faces exude warmth and approachability with soft, welcoming features that create an aura of trustworthiness.",
    color: "from-emerald-500 to-teal-500",
  },
  {
    shape: "Square",
    characteristics: ["Strong jawline", "Defined angles", "Bold structure", "Commanding presence"],
    description:
      "Square faces project strength and determination with architectural precision that speaks to leadership qualities.",
    color: "from-purple-500 to-pink-500",
  },
  {
    shape: "Heart",
    characteristics: ["Wider forehead", "Delicate chin", "Expressive eyes", "Romantic silhouette"],
    description:
      "Heart-shaped faces embody creative elegance with features that taper gracefully from an expressive brow to a refined chin.",
    color: "from-orange-500 to-red-500",
  },
]

export default function FaceShapesGuide() {
  return (
    <div className="relative mb-24">
      <div className="relative bg-white/60 backdrop-blur-xl rounded-3xl overflow-hidden shadow-2xl border border-slate-200">
        {/* Enhanced header */}
        <div className="relative p-12 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-gray-50 overflow-hidden">
          {/* Background pattern */}
          <div className="absolute inset-0 opacity-5">
            <div
              className="absolute inset-0"
              style={{
                backgroundImage: `radial-gradient(circle at 25% 25%, #64748b 2px, transparent 2px),
                               radial-gradient(circle at 75% 75%, #64748b 2px, transparent 2px)`,
                backgroundSize: "60px 60px",
              }}
            ></div>
          </div>

          <div className="relative z-10 text-center">
            <h3 className="text-5xl font-bold text-slate-800 mb-6 hover:scale-105 transition-transform duration-300">
              Face Shape Types
            </h3>
            <p className="text-slate-600 text-xl max-w-3xl mx-auto leading-relaxed">
              Discover the four primary facial structures and their unique characteristics through our advanced AI
              analysis
            </p>
          </div>
        </div>

        <div className="p-12">
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {faceShapes.map((shape, index) => (
              <div key={index} className="group cursor-pointer">
                <div className="relative bg-white/90 backdrop-blur-sm rounded-3xl p-8 border border-slate-200 group-hover:border-slate-300 transition-all duration-500 group-hover:shadow-2xl group-hover:scale-110 overflow-hidden">
                  {/* Animated background gradient */}
                  <div
                    className={`absolute inset-0 bg-gradient-to-br ${shape.color} opacity-0 group-hover:opacity-10 transition-opacity duration-500`}
                  ></div>

                  {/* Floating particles */}
                  <div className="absolute top-4 right-4 w-2 h-2 bg-slate-300 rounded-full opacity-0 group-hover:opacity-100 group-hover:animate-bounce transition-all duration-500"></div>
                  <div
                    className="absolute bottom-4 left-4 w-1.5 h-1.5 bg-slate-400 rounded-full opacity-0 group-hover:opacity-100 group-hover:animate-bounce transition-all duration-500"
                    style={{ animationDelay: "0.2s" }}
                  ></div>

                  <div className="relative z-10">
                    {/* Enhanced icon */}
                    <div
                      className={`w-20 h-20 bg-gradient-to-br ${shape.color} rounded-3xl flex items-center justify-center shadow-xl mx-auto mb-6 group-hover:shadow-2xl group-hover:scale-110 group-hover:rotate-12 transition-all duration-500`}
                    >
                      <User className="w-10 h-10 text-white group-hover:scale-125 transition-transform duration-300" />
                    </div>

                    <h4 className="text-3xl font-bold text-slate-800 mb-4 text-center group-hover:text-slate-900 transition-colors">
                      {shape.shape}
                    </h4>

                    <p className="text-slate-600 text-sm leading-relaxed text-center mb-6 group-hover:text-slate-700 transition-colors">
                      {shape.description.split(".")[0]}.
                    </p>

                    {/* Enhanced characteristics */}
                    <div className="space-y-3">
                      {shape.characteristics.slice(0, 2).map((char, i) => (
                        <div key={i} className="flex items-center justify-center">
                          <span className="px-4 py-2 bg-slate-100 group-hover:bg-slate-50 text-slate-600 group-hover:text-slate-700 rounded-full text-xs font-semibold transition-all duration-300 border border-slate-200 group-hover:border-slate-300">
                            {char}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Shine effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
