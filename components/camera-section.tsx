'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, X } from 'lucide-react';
import { toast } from 'sonner';

interface CameraSectionProps {
  onImageCapture: (file: File) => void;
  onClose: () => void;
}

export function CameraSection({ onImageCapture, onClose }: CameraSectionProps) {
  const [isStreaming, setIsStreaming] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      // Cleanup: stop camera when component unmounts
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      toast.error('Could not access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
  };

  const captureImage = () => {
    if (!videoRef.current) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw the current video frame
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Convert to blob
    canvas.toBlob((blob) => {
      if (!blob) return;
      
      // Create a file from the blob
      const file = new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' });
      onImageCapture(file);
      
      // Stop camera after capture
      stopCamera();
      onClose();
    }, 'image/jpeg', 0.95);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-4 w-full max-w-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Take Photo</h2>
          <Button variant="ghost" size="icon" onClick={() => {
            stopCamera();
            onClose();
          }}>
            <X className="h-5 w-5" />
          </Button>
        </div>

        <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden mb-4">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-full object-cover"
          />
          {!isStreaming && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Button onClick={startCamera} className="gap-2">
                <Camera className="h-5 w-5" />
                Start Camera
              </Button>
            </div>
          )}
        </div>

        {isStreaming && (
          <div className="flex justify-center gap-4">
            <Button onClick={captureImage} className="gap-2">
              <Camera className="h-5 w-5" />
              Capture Photo
            </Button>
            <Button variant="outline" onClick={stopCamera}>
              Stop Camera
            </Button>
          </div>
        )}
      </div>
    </div>
  );
} 