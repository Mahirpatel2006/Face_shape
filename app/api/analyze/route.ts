import { NextResponse } from "next/server"
import { NextRequest } from "next/server"

// This route is now a proxy to the Python serverless function.
// It is good practice to have a defined API route in your Next.js app
// that calls your Python backend. This hides the implementation details
// from the client and provides a single point of entry.

export async function POST(request: NextRequest) {
  try {
    const { imageUrl } = await request.json()

    if (!imageUrl) {
      return NextResponse.json(
        { error: "No imageUrl provided" },
        { status: 400 }
      )
    }

    // IMPORTANT: When deploying to Vercel, Vercel magically knows how to route this.
    // In local development, Next.js will proxy the request to your Python function.
    // The URL must be absolute for the fetch to work correctly in a server component.
    const vercelUrl = process.env.VERCEL_URL
    const baseUrl = vercelUrl ? `https://${vercelUrl}` : 'http://localhost:3000'
    
    const response = await fetch(`${baseUrl}/api/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ imageUrl }),
    })

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json(
        { error: error.error || "Analysis failed in Python function" },
        { status: response.status }
      )
    }

    const result = await response.json()
    return NextResponse.json(result, { status: 200 })
  } catch (error) {
    console.error("Error in Next.js analysis proxy route:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
} 