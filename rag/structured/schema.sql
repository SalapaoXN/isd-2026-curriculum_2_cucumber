PRAGMA foreign_keys = ON;

CREATE TABLE provenance (
    provenance_id INTEGER PRIMARY KEY,
    program TEXT,
    source_filename TEXT,
    source_page INTEGER,
    document_page INTEGER,
    document_category TEXT NOT NULL DEFAULT 'unknown'
        CHECK (document_category IN ('plan', 'description', 'unknown')),
    source_uri TEXT,
    source_locator TEXT,
    excerpt TEXT
);

CREATE TABLE catalogs (
    catalog_id INTEGER PRIMARY KEY,
    catalog_key TEXT,
    title TEXT,
    academic_year TEXT,
    institution TEXT,
    notes TEXT
);

CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id),
    course_code TEXT NOT NULL,
    course_code_normalized TEXT NOT NULL,
    name_th TEXT,
    name_en TEXT,
    credits TEXT,
    credit_units INTEGER,
    credits_raw TEXT,
    description_th TEXT,
    description_en TEXT,
    category TEXT,
    course_type TEXT,
    prerequisite_text TEXT,
    notes TEXT,
    UNIQUE (catalog_id, course_code_normalized)
);

CREATE TABLE curriculum_plans (
    plan_id INTEGER PRIMARY KEY,
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id),
    program_code TEXT NOT NULL,
    plan_code TEXT,
    plan_name TEXT,
    version TEXT,
    notes TEXT
);

CREATE TABLE alternative_course_groups (
    alternative_group_id INTEGER PRIMARY KEY,
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id),
    plan_id INTEGER REFERENCES curriculum_plans(plan_id),
    group_key TEXT,
    label TEXT,
    minimum_choices INTEGER NOT NULL DEFAULT 1 CHECK (minimum_choices >= 1),
    maximum_choices INTEGER NOT NULL DEFAULT 1
        CHECK (maximum_choices >= minimum_choices),
    notes TEXT
);

CREATE TABLE alternative_course_group_members (
    alternative_group_member_id INTEGER PRIMARY KEY,
    alternative_group_id INTEGER NOT NULL
        REFERENCES alternative_course_groups(alternative_group_id)
        ON DELETE CASCADE,
    course_id INTEGER NOT NULL REFERENCES courses(course_id),
    member_order INTEGER,
    notes TEXT,
    UNIQUE (alternative_group_id, course_id)
);

CREATE TABLE plan_placements (
    placement_id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES curriculum_plans(plan_id)
        ON DELETE CASCADE,
    course_id INTEGER REFERENCES courses(course_id),
    alternative_group_id INTEGER
        REFERENCES alternative_course_groups(alternative_group_id),
    year_number INTEGER,
    semester_number INTEGER,
    flexible_year_number INTEGER,
    flexible_semester_number INTEGER,
    flexible_year_semester_raw TEXT,
    category TEXT,
    requirement_type TEXT,
    placement_order INTEGER,
    credits_override TEXT,
    raw_text TEXT,
    notes TEXT,
    CHECK (
        (course_id IS NOT NULL AND alternative_group_id IS NULL)
        OR
        (course_id IS NULL AND alternative_group_id IS NOT NULL)
    )
);

CREATE TABLE prerequisites (
    prerequisite_id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(course_id)
        ON DELETE CASCADE,
    prerequisite_course_id INTEGER REFERENCES courses(course_id),
    alternative_group_id INTEGER
        REFERENCES alternative_course_groups(alternative_group_id),
    prerequisite_order INTEGER,
    requirement_type TEXT NOT NULL DEFAULT 'required',
    raw_text TEXT,
    notes TEXT,
    CHECK (
        (prerequisite_course_id IS NOT NULL AND alternative_group_id IS NULL)
        OR
        (prerequisite_course_id IS NULL AND alternative_group_id IS NOT NULL)
        OR
        (prerequisite_course_id IS NULL AND alternative_group_id IS NULL
            AND raw_text IS NOT NULL)
    )
);

CREATE TABLE catalog_provenance (
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (catalog_id, provenance_id)
);

CREATE TABLE course_provenance (
    course_id INTEGER NOT NULL REFERENCES courses(course_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    source_order INTEGER,
    PRIMARY KEY (course_id, provenance_id)
);

CREATE TABLE curriculum_plan_provenance (
    plan_id INTEGER NOT NULL REFERENCES curriculum_plans(plan_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (plan_id, provenance_id)
);

CREATE TABLE plan_placement_provenance (
    placement_id INTEGER NOT NULL REFERENCES plan_placements(placement_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (placement_id, provenance_id)
);

CREATE TABLE prerequisite_provenance (
    prerequisite_id INTEGER NOT NULL REFERENCES prerequisites(prerequisite_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (prerequisite_id, provenance_id)
);

CREATE TABLE alternative_group_provenance (
    alternative_group_id INTEGER NOT NULL
        REFERENCES alternative_course_groups(alternative_group_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (alternative_group_id, provenance_id)
);

CREATE TABLE alternative_group_member_provenance (
    alternative_group_member_id INTEGER NOT NULL
        REFERENCES alternative_course_group_members(alternative_group_member_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (alternative_group_member_id, provenance_id)
);

CREATE INDEX courses_by_catalog_and_code
    ON courses (catalog_id, course_code);

CREATE INDEX plans_by_catalog
    ON curriculum_plans (catalog_id);

CREATE INDEX placements_by_plan
    ON plan_placements (plan_id, year_number, semester_number, placement_order);

CREATE INDEX placements_by_course
    ON plan_placements (course_id);

CREATE INDEX prerequisites_by_course
    ON prerequisites (course_id, prerequisite_order);

CREATE INDEX alternative_members_by_group
    ON alternative_course_group_members (alternative_group_id, member_order);

CREATE INDEX alternative_members_by_course
    ON alternative_course_group_members (course_id);
