"use client"

import { Brain } from "lucide-react"

export default function Footer() {
  return (
    <footer className="relative">
      <div className="max-w-6xl mx-auto px-6 py-20 text-center">
        <div className="flex justify-center items-center space-x-6 mb-8">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-slate-600 to-gray-700 rounded-2xl blur-lg opacity-50 group-hover:opacity-70 transition-opacity duration-300"></div>
            <div className="relative w-12 h-12 bg-gradient-to-br from-slate-800 to-gray-900 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <Brain className="w-6 h-6 text-white group-hover:rotate-12 transition-transform duration-300" />
            </div>
          </div>
          <span className="text-4xl font-bold text-slate-800 hover:scale-105 transition-transform duration-300">
            FaceSync
          </span>
        </div>

        <p className="text-slate-600 text-xl mb-10 max-w-2xl mx-auto leading-relaxed">
          Advanced facial analysis powered by cutting-edge AI technology.
          <span className="block mt-2 text-slate-500">Discover your unique features with precision and privacy.</span>
        </p>

        <div className="relative inline-block">
          <div className="w-40 h-1 bg-gradient-to-r from-slate-400 to-gray-600 mx-auto rounded-full"></div>
          <div className="absolute inset-0 w-40 h-1 bg-gradient-to-r from-slate-300 to-gray-500 mx-auto rounded-full blur-sm opacity-50"></div>
        </div>
      </div>
    </footer>
  )
}
