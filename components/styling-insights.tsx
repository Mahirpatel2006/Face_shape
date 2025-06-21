"use client"

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Diamond, Glasses, Scissors, VenetianMask, Sparkles, Brush } from 'lucide-react'
import { cn } from '@/lib/utils'

interface StylingInsightsProps {
  faceShape: string
  gender: 'male' | 'female' | 'other'
}

const femaleStylingData = {
    Oval: {
        hairstyles: [
            "Shoulder-length waves or curls",
            "Straight or wavy long hair with layers",
            "Bob cut or lob (long bob)",
            "Side-swept or curtain bangs",
            "Half up, half down hairstyles",
        ],
        glasses: [
            "Almost any shape works well",
            "Try bold, square, or cat-eye frames",
            "Avoid very narrow glasses",
        ],
        styling: [
            "Wear colorful or patterned clothes confidently",
            "Try different necklines—round, V-neck, or square",
            "Earrings of all sizes look good",
            "Use belts or waistbands to show your shape",
        ],
        makeup: [
            "Light contouring to keep face natural",
            "Highlighter on cheekbones and nose",
            "Play with bold lipstick or colorful eyeshadow",
            "Keep brows soft to match natural balance",
        ],
    },
    Round: {
        hairstyles: [
            "Long hair with layers",
            "Side-swept bangs or angled bob",
            "High ponytail or messy bun",
            "Loose waves with middle part",
            "Avoid full, round bobs without layers",
        ],
        glasses: [
            "Square, rectangular, or angular shapes",
            "Cat-eye or upswept frames",
            "Avoid round frames—they make your face look rounder",
        ],
        styling: [
            "V-neck tops make the face look longer",
            "Wear long earrings to pull the look down",
            "Add structure with blazers or shirts with seams",
            "Avoid round necklines or choker necklaces",
        ],
        makeup: [
            "Contour sides of cheeks and jawline",
            "Use blush slightly higher than cheekbone",
            "Highlight down the nose and forehead center",
            "Define brows to add structure",
        ],
    },
    Square: {
        hairstyles: [
            "Soft waves or big curls",
            "Side part with long layers",
            "Loose ponytail with volume",
            "Long bob (lob) with soft ends",
            "Avoid very straight hair with no volume",
        ],
        glasses: [
            "Round or oval glasses",
            "Thin or rimless frames",
            "Avoid sharp or square glasses",
        ],
        styling: [
            "Scoop necks, off-shoulder, or curved collars",
            "Flowy dresses or skirts",
            "Layered jewelry to soften the look",
            "Avoid square shoulder pads or stiff fabrics",
        ],
        makeup: [
            "Contour jawline lightly",
            "Blush in round motion on cheeks",
            "Highlight center of forehead and chin",
            "Soft brows look better than sharp ones",
        ],
    },
    Heart: {
        hairstyles: [
            "Chin-length or shoulder bobs",
            "Side part with soft waves",
            "Curly styles or volume near jaw",
            "Bangs to cover forehead",
            "Avoid styles with volume only at the crown",
        ],
        glasses: [
            "Bottom-heavy or rounded frames",
            "Light-colored or rimless glasses",
            "Avoid wide or bold top-heavy frames",
        ],
        styling: [
            "Wear A-line dresses or wide-leg pants",
            "Keep tops simple and avoid shoulder pads",
            "Short necklaces or chokers balance the neck",
        ],
        makeup: [
            "Contour temples to reduce forehead width",
            "Highlight cheeks and chin",
            "Light eye makeup with longer lashes",
            "Keep lips defined to draw attention downward",
        ],
    },
    Oblong: {
        hairstyles: [
            "**Bangs are Your Best Friend:** Blunt, wispy, or curtain bangs can visually shorten the face.",
            "**Chin-Length Bob:** A bob that ends at the chin can add the illusion of width.",
            "**Curls and Waves:** Volume on the sides of the head is key to balancing the length of the face.",
            "**Shoulder-Length Cuts:** Opt for styles that are around shoulder length with layers to add body.",
            "**Avoid:** Very long, straight hair or styles with a lot of height at the crown, which will only elongate the face."
        ],
        glasses: [
            "**Wide Frames:** Look for frames that are wider than they are tall, such as oversized or square shapes.",
            "**Decorative Temples:** Bold details on the arms of the glasses can help to add width to the face.",
            "**Low Bridge:** A lower bridge can help to make the nose appear shorter.",
            "**Avoid:** Narrow or small frames that will look out of proportion."
        ],
        styling: [], // Placeholder for Oblong female styling
        makeup: [
            "**Shorten the Face:** Apply contour along the hairline and under the chin to create the illusion of a shorter face.",
            "**Widen the Center:** Apply blush on the apples of the cheeks and blend it outwards towards the ears.",
            "**Horizontal Highlighting:** When highlighting, think horizontally. Apply it on the cheekbones but don't extend it too high.",
            "**Avoid Vertical Lines:** All makeup application should focus on creating width rather than length."
        ]
    }
};

const maleStylingData = {
    Oval: {
        hairstyles: [
            "Classic short fade or crew cut",
            "Side part with a clean finish",
            "Slicked back or combed over",
            "Buzz cut or textured top",
            "Medium length with soft layers",
        ],
        glasses: [
            "Square or rounded frames",
            "Bold or clear frames work great",
        ],
        styling: [
            "T-shirts, polos, and collars all work well",
            "Blazers or jackets with fitted waist look sharp",
            "Try sneakers or loafers depending on the style",
        ],
        beard: [
            "Clean shave or short stubble looks great",
            "Short boxed beard or goatee also works",
            "Avoid long or wide beards—they may hide your natural balance",
        ],
    },
    Round: {
        hairstyles: [
            "Pompadour or faux hawk",
            "High fade with volume on top",
            "Textured crop with spiky finish",
            "Side part with short sides",
            "Avoid flat haircuts with no height",
        ],
        glasses: [
            "Rectangle or square frames",
            "Avoid rounded or small frames",
        ],
        styling: [
            "Wear V-neck or open collars",
            "Use vertical stripes or seams",
            "Layered jackets or shirts help define shape",
        ],
        beard: [
            "Goatee or short pointed beard adds length",
            "Chinstrap beard sharpens jawline",
            "Avoid wide or thick full beards",
        ],
    },
    Square: {
        hairstyles: [
            "Side part with soft texture",
            "Messy top with faded sides",
            "Medium quiff or fringe",
            "Short back and sides with volume on top",
            "Avoid styles that make head look boxy",
        ],
        glasses: [
            "Round, curved, or oval frames",
            "Avoid thick, square glasses",
        ],
        styling: [
            "Wear round-neck or soft-collar shirts",
            "Add movement with layered clothing",
            "Avoid stiff jackets or boxy fits",
        ],
        beard: [
            "Round or faded beard shapes",
            "Short full beard with soft edge",
            "Avoid very sharp or blocky beards",
        ],
    },
    Heart: {
        hairstyles: [
            "Messy top with faded sides",
            "Textured fringe or side-swept front",
            "Mid-length with volume near the jawline",
            "Avoid high volume at the top only",
        ],
        glasses: [
            "Oval or bottom-heavy frames",
            "Avoid glasses that make forehead look bigger",
        ],
        styling: [
            "Keep upper body simple—no heavy layers or collars",
            "Use dark colors on top and lighter colors below",
            "Straight-leg or wide-leg pants add balance",
        ],
        beard: [
            "Full beard or rounded beard adds weight to jaw",
            "Avoid pointy chin styles like a sharp goatee",
        ],
    },
    Oblong: {
        hairstyles: [
            "**Width and Volume:** Styles that add volume on the sides, such as a textured crop or a longer, layered cut, are ideal.",
            "**Fringe:** A fringe, either straight across or side-swept, can significantly shorten the appearance of the face.",
            "**Avoid:** Styles with excessive height on top (like a high pompadour) or very short sides, as this will further elongate the face.",
            "**Side Part:** A side part can add some width and break up the vertical lines."
        ],
        glasses: [
            "**Tall Frames:** Frames with more depth than width can help make the face appear shorter and more balanced.",
            "**Decorative Temples:** Embellishments or bold colors on the temples can add width to the face.",
            "**Low Bridge:** A low bridge on the glasses can shorten the nose, contributing to a more balanced look.",
            "**Avoid:** Small, narrow frames that will be out of proportion and emphasize the length of the face."
        ],
        styling: [], // Placeholder for Oblong male styling
        beard: [
            "**Fuller on the Sides:** A beard that is fuller on the cheeks can add width to the face, creating a more balanced appearance.",
            "**Mutton Chops or a Wide Goatee:** These styles can help to visually widen the jawline.",
            "**Avoid:** Beards that are long and pointy at the chin (like a Van Dyke or a long goatee), as this will exaggerate the face's length.",
            "**Stubble:** A well-maintained, full stubble can add texture without adding too much length."
        ]
    }
};

const stylingData = {
    male: maleStylingData,
    female: femaleStylingData,
    other: {
        Round: { ...maleStylingData.Round, makeup: femaleStylingData.Round.makeup },
        Square: { ...maleStylingData.Square, makeup: femaleStylingData.Square.makeup },
        Oval: { ...maleStylingData.Oval, makeup: femaleStylingData.Oval.makeup },
        Heart: { ...maleStylingData.Heart, makeup: femaleStylingData.Heart.makeup },
        Oblong: { ...maleStylingData.Oblong, makeup: femaleStylingData.Oblong.makeup },
    }
}

type FaceShape = keyof (typeof stylingData.male);
type Category = 'hairstyles' | 'glasses' | 'beard' | 'makeup' | 'styling';

const categoryIcons: Record<Category, React.ElementType> = {
  hairstyles: Scissors,
  glasses: Glasses,
  beard: VenetianMask,
  makeup: Brush,
  styling: Sparkles
}

const categoryLabels: Record<Category, string> = {
    hairstyles: 'Hairstyles',
    glasses: 'Eyewear',
    beard: 'Beard Styles',
    makeup: 'Makeup',
    styling: 'Styling'
}

const InsightItem = ({ text, delay }: { text: string; delay: number }) => {
    const parts = text.split('**');
    return (
        <motion.li
            className="flex items-start space-x-3 p-4 rounded-lg bg-slate-100/60 hover:bg-white/80 transition-all duration-300 shadow"
            variants={{
                hidden: { opacity: 0, y: 20 },
                visible: { opacity: 1, y: 0, transition: { delay } }
            }}
        >
            <div className="w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 mt-2 flex-shrink-0" />
            <p className="text-slate-700">
                {parts.map((part, i) =>
                    i % 2 === 1 ? <strong key={i} className="font-semibold text-slate-900">{part}</strong> : part
                )}
            </p>
        </motion.li>
    )
}

export default function StylingInsights({ faceShape, gender }: StylingInsightsProps) {
    const [activeCategory, setActiveCategory] = useState<Category>('hairstyles')

    if (!faceShape || !gender) {
        return (
            <div className="min-h-[300px] flex items-center justify-center bg-white/50 backdrop-blur-xl rounded-3xl shadow-lg border border-slate-200">
                <p className="text-slate-500 text-lg">Select a gender to see styling insights.</p>
            </div>
        )
    }

    const safeFaceShape = faceShape as FaceShape
    const insights = stylingData[gender === 'female' ? 'female' : 'male'][safeFaceShape]
    
    const categoriesToShow: Category[] = gender === 'female' 
      ? ['hairstyles', 'glasses', 'styling', 'makeup']
      : ['hairstyles', 'glasses', 'styling', 'beard'];


    const handleCategoryClick = (category: Category) => {
        setActiveCategory(category)
    }

    return (
        <div className="relative bg-white/70 backdrop-blur-xl rounded-3xl overflow-hidden shadow-2xl border border-slate-200 animate-fade-in">
            <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-b from-white/50 to-transparent z-0" />
            <div className="relative z-10 p-8">
                <div className="text-center mb-10">
                    <h2 className="text-4xl font-extrabold text-slate-800 tracking-tight">Style Guide</h2>
                    <p className="text-lg text-slate-600 mt-1">
                        Recommendations for your <span className="font-bold text-purple-600">{faceShape}</span> face shape.
                    </p>
                </div>
                
                <div className="flex flex-col md:flex-row gap-8">
                    {/* Vertical Navigation */}
                    <nav className="flex md:flex-col gap-2">
                        {categoriesToShow.map((cat) => {
                            const Icon = categoryIcons[cat]
                            const isActive = activeCategory === cat
                            return (
                                <button
                                    key={cat}
                                    onClick={() => handleCategoryClick(cat)}
                                    className={cn(
                                        "flex items-center justify-start text-left w-full md:w-48 p-3 rounded-lg text-lg font-semibold transition-all duration-300 ease-in-out transform",
                                        "focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-white/80",
                                        isActive 
                                            ? 'bg-gradient-to-r from-slate-800 to-slate-900 text-white shadow-lg scale-105' 
                                            : 'text-slate-600 hover:bg-slate-200/70 hover:text-slate-800'
                                    )}
                                >
                                    <Icon className="w-5 h-5 mr-3 flex-shrink-0" />
                                    {categoryLabels[cat]}
                                </button>
                            )
                        })}
                    </nav>

                    {/* Content */}
                    <div className="flex-1 bg-slate-50/50 rounded-2xl p-6 min-h-[300px]">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={activeCategory}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3, ease: 'easeInOut' }}
                            >
                                <motion.ul 
                                  className="space-y-4"
                                  variants={{
                                    visible: { transition: { staggerChildren: 0.1 } },
                                    hidden: {}
                                  }}
                                  initial="hidden"
                                  animate="visible"
                                >
                                    {(insights[activeCategory as keyof typeof insights] as string[] || []).map((tip, index) => (
                                        <InsightItem key={index} text={tip} delay={(index + 1) * 0.1} />
                                    ))}
                                </motion.ul>
                            </motion.div>
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    )
} 