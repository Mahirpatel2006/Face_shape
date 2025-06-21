// app\api\upload\route.ts

import { put } from '@vercel/blob';
import { NextResponse } from 'next/server';

export async function POST(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const filename = searchParams.get('filename');

  if (!filename || !request.body) {
    return NextResponse.json(
      { error: 'No filename or file body provided.' },
      { status: 400 },
    );
  }

  try {
    // The request body is the file itself.
    // The 'put' function from @vercel/blob handles the streaming upload.
    const blob = await put(filename, request.body, {
      access: 'public',
      token: process.env.BLOB_READ_WRITE_TOKEN, // Make sure to set this env variable
    });

    // The 'blob' object contains the URL of the uploaded file.
    return NextResponse.json(blob);
  } catch (error) {
    console.error('Error uploading to Vercel Blob:', error);
    const errorMessage = (error instanceof Error) ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: 'Error uploading file.', details: errorMessage },
      { status: 500 },
    );
  }
} 