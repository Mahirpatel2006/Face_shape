"use client"

import type React from "react"
import { useCallback, useState } from "react"
import { Upload, Camera, User, Check } from "lucide-react"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { CameraSection } from "./camera-section"
import { toast } from "sonner"
import { cn } from "@/lib/utils"

interface UploadSectionProps {
  onImageUpload: (imageUrl: string) => void
  onAnalyze: (imageUrl: string, gender: string) => void
  isAnalyzing: boolean
  gender: string
  setGender: (gender: string) => void
}

export default function UploadSection({
  onImageUpload,
  onAnalyze,
  isAnalyzing,
  gender,
  setGender,
}: UploadSectionProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [showCamera, setShowCamera] = useState(false)

  const handleFileUpload = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file")
      return
    }

    setIsUploading(true)
    
    try {
      const response = await fetch(`/api/upload?filename=${file.name}`, {
        method: 'POST',
        body: file,
      })

      const blob = await response.json()

      if (!response.ok) {
        throw new Error(blob.error || 'Failed to upload image')
      }

      toast.success('Image uploaded successfully')
      onImageUpload(blob.url)
      onAnalyze(blob.url, gender)
    } catch (error) {
      console.error('Upload error:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to upload image')
    } finally {
      setIsUploading(false)
    }
  }

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        handleFileUpload(acceptedFiles[0])
      }
    },
    [gender] // Dependency on gender ensures the latest value is used
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    maxFiles: 1,
    disabled: isUploading || isAnalyzing,
  })

  const handleCameraCapture = (file: File) => {
    setShowCamera(false)
    handleFileUpload(file)
  }

  return (
    <div className="w-full max-w-2xl mx-auto p-4">
      <div className="flex justify-center items-center space-x-4 mb-8">
        <span className="text-slate-700 font-semibold">I am:</span>
        <div
          className={cn(
            "flex items-center space-x-2 rounded-full p-1.5 transition-all duration-300",
            gender === "female" ? "bg-pink-100" : "bg-blue-100"
          )}
        >
          <Button
            onClick={() => setGender("female")}
            variant="ghost"
            className={cn(
              "rounded-full px-6 py-2 transition-all duration-300",
              gender === "female" && "bg-white shadow-md text-pink-600"
            )}
          >
            Female
          </Button>
          <Button
            onClick={() => setGender("male")}
            variant="ghost"
            className={cn(
              "rounded-full px-6 py-2 transition-all duration-300",
              gender === "male" && "bg-white shadow-md text-blue-600"
            )}
          >
            Male
          </Button>
        </div>
      </div>

      <div
        {...getRootProps()}
        className={cn(
          "relative border-2 border-dashed rounded-3xl p-8 text-center cursor-pointer transition-all duration-300 group",
          "hover:border-blue-500 hover:bg-blue-50/50",
          isDragActive && "border-solid border-blue-600 bg-blue-100",
          (isUploading || isAnalyzing) && "opacity-50 cursor-not-allowed"
        )}
      >
        <input {...getInputProps()} />
        <div
          className={cn(
            "absolute inset-0 bg-gradient-to-br from-white via-blue-50 to-purple-50 scale-95 opacity-0 transition-all duration-300 rounded-3xl",
            "group-hover:scale-100 group-hover:opacity-100"
          )}
        />
        <div className="relative z-10 flex flex-col items-center justify-center space-y-4 h-48">
          <div
            className={cn(
              "w-20 h-20 rounded-full bg-slate-100 flex items-center justify-center transition-all duration-300",
              "group-hover:bg-white group-hover:scale-110 group-hover:shadow-lg"
            )}
          >
            <Upload className="h-10 w-10 text-slate-400 group-hover:text-blue-500 transition-colors duration-300" />
          </div>
          <div>
            <p className="font-semibold text-slate-700 mb-1">
              {isUploading
                ? "Uploading..."
                : isAnalyzing
                ? "Analyzing..."
                : isDragActive
                ? "Drop image to upload!"
                : "Drag & drop image here"}
            </p>
            <p className="text-sm text-slate-500">or click to select a file</p>
          </div>
        </div>
      </div>

      <div className="flex items-center my-6">
        <div className="flex-grow border-t border-slate-300"></div>
        <span className="flex-shrink mx-4 text-slate-500 text-sm">OR</span>
        <div className="flex-grow border-t border-slate-300"></div>
      </div>

      <div className="text-center">
        <Button
          onClick={() => setShowCamera(true)}
          disabled={isUploading || isAnalyzing}
          className="rounded-full bg-slate-800 text-white px-8 py-6 text-lg font-semibold hover:bg-slate-900 hover:shadow-lg transition-all duration-300"
        >
          <Camera className="h-6 w-6 mr-3" />
          Take a Photo
        </Button>
      </div>

      {showCamera && (
        <CameraSection onImageCapture={handleCameraCapture} onClose={() => setShowCamera(false)} />
      )}
    </div>
  )
}
